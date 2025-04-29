import os
import re
import tempfile
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from img2table.document import Image as Img2TableImage
from img2table.ocr import TesseractOCR
import cv2
import numpy as np

def parse_reference_range(ref_range):
    match = re.match(r"([\d.]+)\s*-\s*([\d.]+)", ref_range)
    if match:
        return float(match.group(1)), float(match.group(2))
    return None, None

def is_out_of_range(value, ref_range):
    try:
        low, high = parse_reference_range(ref_range)
        value = float(value)
        if low is not None and high is not None:
            return value < low or value > high
    except Exception:
        pass
    return None

class LabTest(BaseModel):
    test_name: str
    test_value: str
    bio_reference_range: str
    test_unit: str
    lab_test_out_of_range: bool

class LabTestResponse(BaseModel):
    is_success: bool
    data: List[LabTest]

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

@app.post("/get-lab-tests", response_model=LabTestResponse)
async def get_lab_tests(file: UploadFile = File(...)):
    try:
        print(f"Processing file: {file.filename}")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            content = await file.read()
            print(f"File size: {len(content)} bytes")
            tmp.write(content)
            tmp_path = tmp.name
            print(f"Temporary file created at: {tmp_path}")


        print("Preprocessing image...")
        img = cv2.imread(tmp_path)
        if img is None:
            print("Error: Could not read image file")
            return LabTestResponse(is_success=False, data=[])
            

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        

        gray = cv2.bilateralFilter(gray, 9, 75, 75)
        

        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        

        kernel = np.ones((1,1), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        

        kernel = np.ones((2,2), np.uint8)
        binary = cv2.dilate(binary, kernel, iterations=1)
        

        preprocessed_path = tmp_path.replace('.png', '_preprocessed.png')
        cv2.imwrite(preprocessed_path, binary)
        print(f"Preprocessed image saved at: {preprocessed_path}")

        print("Initializing image processing...")
        img = Img2TableImage(src=preprocessed_path)
        

        ocr = TesseractOCR(
            lang="eng",
            psm=6  
        )
        
        print("Extracting tables...")
        tables = img.extract_tables(
            ocr=ocr,
            implicit_rows=True,
            borderless_tables=True,
            min_confidence=30  
        )
        print(f"Found {len(tables)} tables in the image")

        lab_tests = []
        
        for table_idx, table in enumerate(tables):
            print(f"\nProcessing table {table_idx + 1}")
            df = table.df
            
            df = df.replace('None', np.nan)
            df = df.replace('', np.nan)
            df = df.dropna(how='all')
            df = df.dropna(axis=1, how='all')
            
            print(f"Table shape after cleaning: {df.shape}")
            if df.shape[0] < 2 or df.shape[1] < 2:
                print("Skipping table - too small after cleaning")
                continue
                
            print("Columns:", df.columns.tolist())
            print("First few rows of data:")
            print(df.head())
            
            col_map = {}
            for col in df.columns:
                col_lower = str(col).lower()
                print(f"Checking column: {col} ({col_lower})")
                
                sample_values = df[col].dropna().astype(str).tolist()
                if not sample_values:
                    continue
                    
                print(f"Sample values: {sample_values}")
                

                if any(any(word in val.lower() for word in ['test', 'parameter', 'investigation', 'rbc', 'wbc', 'hb', 'platelet']) for val in sample_values):
                    col_map['test_name'] = col

                elif any(re.match(r'^\d*\.?\d+\s*[a-zA-Z/]*$', val.strip()) for val in sample_values):
                    col_map['test_value'] = col

                elif any(re.match(r'[\d.]+[\s-]+[\d.]+', val.strip()) for val in sample_values):
                    col_map['bio_reference_range'] = col

                elif any(any(unit in val.lower() for unit in ['g/dl', 'mg/dl', 'iu/l', 'mmol/l', '/cumm', '%', 'fl', 'pg']) for val in sample_values):
                    col_map['test_unit'] = col

                elif "test" in col_lower and ("name" in col_lower or "parameter" in col_lower):
                    col_map['test_name'] = col
                elif "value" in col_lower or "result" in col_lower:
                    col_map['test_value'] = col
                elif "range" in col_lower or "reference" in col_lower:
                    col_map['bio_reference_range'] = col
                elif "unit" in col_lower:
                    col_map['test_unit'] = col
            
            print("Column mapping:", col_map)
            
            if len(col_map) < 2:
                print("Skipping table - insufficient column mapping")
                continue
            
            columns = list(df.columns)
            if 'test_name' not in col_map and len(columns) > 0:
                col_map['test_name'] = columns[0]
            if 'test_value' not in col_map and len(columns) > 1:
                col_map['test_value'] = columns[1]
            if 'bio_reference_range' not in col_map and len(columns) > 2:
                col_map['bio_reference_range'] = columns[2]
            if 'test_unit' not in col_map and len(columns) > 3:
                col_map['test_unit'] = columns[3]
            
            for row_idx, (_, row) in enumerate(df.iterrows()):
                try:
                    test_name = str(row.get(col_map.get('test_name', ''), '')).strip()
                    test_value = str(row.get(col_map.get('test_value', ''), '')).strip()
                    bio_reference_range = str(row.get(col_map.get('bio_reference_range', ''), '')).strip()
                    test_unit = str(row.get(col_map.get('test_unit', ''), '')).strip()
                    
                    if not test_name or test_name.lower() in ['none', 'nan', '']:
                        continue
                    if not test_value or test_value.lower() in ['none', 'nan', '']:
                        continue
                    
                    if not test_unit and '/' in test_value:
                        parts = test_value.split()
                        if len(parts) > 1:
                            test_value = parts[0]
                            test_unit = ' '.join(parts[1:])
                    
                    print(f"\nRow {row_idx + 1}:")
                    print(f"Test Name: {test_name}")
                    print(f"Test Value: {test_value}")
                    print(f"Reference Range: {bio_reference_range}")
                    print(f"Unit: {test_unit}")
                    
                    out_of_range = is_out_of_range(test_value, bio_reference_range)
                    if out_of_range is None:
                        out_of_range = False
                    lab_tests.append(LabTest(
                        test_name=test_name,
                        test_value=test_value,
                        bio_reference_range=bio_reference_range,
                        test_unit=test_unit,
                        lab_test_out_of_range=out_of_range
                    ))
                except Exception as e:
                    print(f"Error processing row {row_idx + 1}: {str(e)}")
                    continue
        
        os.remove(tmp_path)
        os.remove(preprocessed_path)
        print(f"\nTotal lab tests extracted: {len(lab_tests)}")
        return LabTestResponse(is_success=True, data=lab_tests)
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return LabTestResponse(is_success=False, data=[]) 