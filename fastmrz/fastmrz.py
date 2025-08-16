import numpy as np
import cv2
import pytesseract
import os
import base64
from datetime import datetime
from io import BytesIO
from PIL import Image

class FastMRZ:
    def __init__(self, tesseract_path="", lang=""):
        self.tesseract_path = tesseract_path
        self.lang = lang
        self.net = cv2.dnn.readNetFromONNX(
            os.path.join(os.path.dirname(__file__), "model/mrz_seg.onnx")
        )

    def _process_image(self, image_path):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR) if isinstance(image_path, str) else image_path

        image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_NEAREST)
        image = np.asarray(np.float32(image / 255))

        if len(image.shape) >= 3:
            image = image[:, :, :3]
        image = np.reshape(image, (1, 256, 256, 3))

        return image

    def _get_roi(self, output_data, image_path):
        if self.tesseract_path != "":
            pytesseract.pytesseract.tesseract_cmd = self.tesseract_path
        image = cv2.imread(image_path, cv2.IMREAD_COLOR) if isinstance(image_path, str) else image_path

        output_data = (output_data[0, :, :, 0] > 0.25) * 1
        output_data = np.uint8(output_data * 255)
        altered_image = cv2.resize(output_data, (image.shape[1], image.shape[0]))

        kernel = np.ones((5, 5), dtype=np.float32)
        altered_image = cv2.erode(altered_image, kernel, iterations=1)

        contours, hierarchy = cv2.findContours(altered_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if len(contours) == 0:
            return ""

        c_area = np.zeros([len(contours)])
        for j in range(len(contours)):
            c_area[j] = cv2.contourArea(contours[j])

        x, y, w, h = cv2.boundingRect(contours[np.argmax(c_area)])

        # Add padding to the ROI
        padding = 10
        x_start = max(0, x - padding)
        y_start = max(0, y - padding)
        x_end = min(image.shape[1], x + w + padding)
        y_end = min(image.shape[0], y + h + padding)

        roi_arr = image[y_start:y_end, x_start:x_end].copy()
        # roi_arr = cv2.convertScaleAbs(roi_arr, alpha=1.25, beta=-50)

        # Apply additional preprocessing to ROI before OCR
        roi_gray = cv2.cvtColor(roi_arr, cv2.COLOR_BGR2GRAY)

        # kernel = np.ones((1, 1), np.uint8)
        # roi_dilate = cv2.dilate(roi_gray, kernel, iterations=1)
        # roi_dilate = cv2.erode(roi_dilate, kernel, iterations=1)

        roi_threshold = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        # Configure pytesseract parameters for better MRZ recognition
        custom_config = r'--oem 3 --psm 6'
        return pytesseract.image_to_string(roi_threshold, lang=self.lang, config=custom_config)

    def _cleanse_roi(self, mrz_text):
        input_list = mrz_text.replace(" ", "").split("\n")

        selection_length = next((len(item) for item in input_list if "<" in item and len(item) in {30, 36, 44}), None,)
        if selection_length is None:
            return ""
        new_list = [item for item in input_list if len(item) >= selection_length]

        return "\n".join(new_list)

    def _get_final_checkdigit(self, input_string, input_type):
        if input_type == "TD3":
            return self._get_checkdigit(input_string[1][:10] + input_string[1][13:20] + input_string[1][21:43])
        elif input_type == "TD2":
            return self._get_checkdigit(input_string[1][:10] + input_string[1][13:20] + input_string[1][21:35])
        else:
            return self._get_checkdigit(input_string[0][5:] + input_string[1][:7] + input_string[1][8:15]
                                         + input_string[1][18:29])

    def _get_checkdigit(self, input_string):
        weights_pattern = [7, 3, 1]

        total = 0
        for i, char in enumerate(input_string):
            if char.isdigit():
                value = int(char)
            elif char.isalpha():
                value = ord(char.upper()) - ord("A") + 10
            else:
                value = 0
            total += value * weights_pattern[i % len(weights_pattern)]

        check_digit = total % 10

        return str(check_digit)

    def _format_date(self, input_date):
        formatted_date = str(datetime.strptime(input_date, "%y%m%d").date())

        return formatted_date

    def _get_birth_date(self, birth_date_str, expiry_date_str):
        birth_year = int(birth_date_str[:4])
        expiry_year = int(expiry_date_str[:4])

        if expiry_year > birth_year:
            return birth_date_str
        adjusted_year = birth_year - 100

        return f"{adjusted_year}-{birth_date_str[5:]}"

    def _is_valid(self, image):
        if isinstance(image, str):
            return bool(os.path.isfile(image))
        elif isinstance(image, np.ndarray):
            return image.shape[-1] == 3

    def _get_mrz(self, image):
        image_array = self._process_image(image)
        self.net.setInput(image_array)
        output_data = self.net.forward()
        mrz_roi = self._get_roi(output_data, image)

        return self._cleanse_roi(mrz_roi)

    def _image_to_base64(self, imagepath):
        image_file = open(imagepath, "rb")
        image_data = image_file.read()
        image_file.close()
        base64_string = base64.b64encode(image_data).decode("utf-8")

        return base64_string

    def _base64_to_array(self, base64_string):
        image_data = base64.b64decode(base64_string)
        image_stream = BytesIO(image_data)
        image = Image.open(image_stream)
        image_array = np.array(image)

        return image_array

    def _parse_mrz(self, mrz_text, include_checkdigit=True):
        if not mrz_text:
            return {"status": "FAILURE", "status_message": "No MRZ detected"}
        mrz_lines = mrz_text.strip().split("\n")
        if len(mrz_lines) not in [2, 3]:
            return {"status": "FAILURE", "status_message": "Invalid MRZ format"}

        # Define relevant fields that we actually need
        relevant_fields = [
            'surname',
            'given_name', 
            'document_number',
            'birth_date',
            'sex',
            'issue_date',
            'expiry_date',
            'nationality_code'
        ]
        
        mrz_code_dict = {}
        checksum_failures = []  # Track checksum failures separately
        
        if len(mrz_lines) == 2:
            # Determine MRZ type based on length and last character
            if mrz_lines[1][-1] == '<':
                mrz_code_dict["mrz_type"] = "MRVB" if len(mrz_lines[0]) == 36 else "MRVA"
            else:
                mrz_code_dict["mrz_type"] = "TD2" if len(mrz_lines[0]) == 36 else "TD3"

            # Line 1 - Common for all 2-line formats
            try:
                mrz_code_dict["document_code"] = mrz_lines[0][:2].strip("<")
            except:
                mrz_code_dict["document_code"] = ""

            try:
                mrz_code_dict["issuer_code"] = mrz_lines[0][2:5]
                if not mrz_code_dict["issuer_code"].isalpha():
                    mrz_code_dict["issuer_code"] = ""
            except:
                mrz_code_dict["issuer_code"] = ""

            # Extract names with error handling
            try:
                names = mrz_lines[0][5:].split("<<")
                mrz_code_dict["surname"] = names[0].replace("<", " ").strip() if len(names) > 0 else ""
                mrz_code_dict["given_name"] = names[1].replace("<", " ").strip() if len(names) > 1 else ""
            except:
                mrz_code_dict["surname"] = ""
                mrz_code_dict["given_name"] = ""

            # Line 2 - Format specific parsing with error handling
            if mrz_code_dict["mrz_type"] in ["TD2", "TD3", "MRVA", "MRVB"]:
                # Document number
                try:
                    mrz_code_dict["document_number"] = mrz_lines[1][:9].replace("<", "").strip()
                    if len(mrz_lines[1]) > 9:
                        document_number_checkdigit = self._get_checkdigit(mrz_code_dict["document_number"])
                        if document_number_checkdigit != mrz_lines[1][9]:
                            checksum_failures.append("Document number checksum mismatch")
                        if include_checkdigit:
                            mrz_code_dict["document_number_checkdigit"] = document_number_checkdigit
                except:
                    mrz_code_dict["document_number"] = ""

                # Nationality
                try:
                    mrz_code_dict["nationality_code"] = mrz_lines[1][10:13]
                    if not mrz_code_dict["nationality_code"].isalpha():
                        mrz_code_dict["nationality_code"] = ""
                except:
                    mrz_code_dict["nationality_code"] = ""

                # Birth date
                try:
                    mrz_code_dict["birth_date"] = mrz_lines[1][13:19]
                    if len(mrz_lines[1]) > 19:
                        birth_date_checkdigit = self._get_checkdigit(mrz_code_dict["birth_date"])
                        if birth_date_checkdigit != mrz_lines[1][19]:
                            checksum_failures.append("Birth date checksum mismatch")
                        if include_checkdigit:
                            mrz_code_dict["birth_date_checkdigit"] = birth_date_checkdigit
                    mrz_code_dict["birth_date"] = self._format_date(mrz_code_dict["birth_date"])
                except:
                    mrz_code_dict["birth_date"] = ""

                # Sex
                try:
                    mrz_code_dict["sex"] = mrz_lines[1][20] if len(mrz_lines[1]) > 20 else ""
                except:
                    mrz_code_dict["sex"] = ""

                # Expiry date
                try:
                    mrz_code_dict["expiry_date"] = mrz_lines[1][21:27]
                    if len(mrz_lines[1]) > 27:
                        expiry_date_checkdigit = self._get_checkdigit(mrz_code_dict["expiry_date"])
                        if expiry_date_checkdigit != mrz_lines[1][27]:
                            checksum_failures.append("Expiry date checksum mismatch")
                        if include_checkdigit:
                            mrz_code_dict["expiry_date_checkdigit"] = expiry_date_checkdigit
                    mrz_code_dict["expiry_date"] = self._format_date(mrz_code_dict["expiry_date"])
                    
                    # Calculate birth date with expiry date context
                    if mrz_code_dict["birth_date"] and mrz_code_dict["expiry_date"]:
                        mrz_code_dict["birth_date"] = self._get_birth_date(mrz_code_dict["birth_date"], mrz_code_dict["expiry_date"])
                except:
                    mrz_code_dict["expiry_date"] = ""

                # Optional data (non-critical)
                try:
                    if mrz_code_dict["mrz_type"] == "TD2":
                        mrz_code_dict["optional_data"] = mrz_lines[1][28:35].strip("<")
                    elif mrz_code_dict["mrz_type"] == "TD3":
                        mrz_code_dict["optional_data"] = mrz_lines[1][28:42].strip("<")
                    elif mrz_code_dict["mrz_type"] == "MRVA":
                        mrz_code_dict["optional_data"] = mrz_lines[1][28:43].strip("<")
                    else:  # MRVB
                        mrz_code_dict["optional_data"] = mrz_lines[1][28:35].strip("<")
                except:
                    mrz_code_dict["optional_data"] = ""

            # Final checkdigit validation (non-critical for visa formats)
            try:
                if mrz_code_dict["mrz_type"] not in ("MRVA", "MRVB"):
                    final_checkdigit = self._get_final_checkdigit(mrz_lines, mrz_code_dict["mrz_type"])
                    if mrz_lines[1][-1] != final_checkdigit:
                        checksum_failures.append("Final checksum mismatch")
                    if include_checkdigit:
                        mrz_code_dict["final_checkdigit"] = final_checkdigit
            except:
                pass  # Don't fail on final checkdigit errors

        else:
            # TD1 format (3 lines)
            mrz_code_dict["mrz_type"] = "TD1"

            # Line 1
            try:
                mrz_code_dict["document_code"] = mrz_lines[0][:2].strip("<")
            except:
                mrz_code_dict["document_code"] = ""

            try:
                mrz_code_dict["issuer_code"] = mrz_lines[0][2:5]
                if not mrz_code_dict["issuer_code"].isalpha():
                    mrz_code_dict["issuer_code"] = ""
            except:
                mrz_code_dict["issuer_code"] = ""

            try:
                mrz_code_dict["document_number"] = mrz_lines[0][5:14].replace("<", "").strip()
                if len(mrz_lines[0]) > 14:
                    document_number_checkdigit = self._get_checkdigit(mrz_code_dict["document_number"])
                    if document_number_checkdigit != mrz_lines[0][14]:
                        checksum_failures.append("Document number checksum mismatch")
                    if include_checkdigit:
                        mrz_code_dict["document_number_checkdigit"] = document_number_checkdigit
            except:
                mrz_code_dict["document_number"] = ""

            try:
                mrz_code_dict["optional_data_1"] = mrz_lines[0][15:].strip("<")
            except:
                mrz_code_dict["optional_data_1"] = ""

            # Line 2
            try:
                mrz_code_dict["birth_date"] = mrz_lines[1][:6]
                if len(mrz_lines[1]) > 6:
                    birth_date_checkdigit = self._get_checkdigit(mrz_code_dict["birth_date"])
                    if birth_date_checkdigit != mrz_lines[1][6]:
                        checksum_failures.append("Birth date checksum mismatch")
                    if include_checkdigit:
                        mrz_code_dict["birth_date_checkdigit"] = birth_date_checkdigit
                mrz_code_dict["birth_date"] = self._format_date(mrz_code_dict["birth_date"])
            except:
                mrz_code_dict["birth_date"] = ""

            try:
                mrz_code_dict["sex"] = mrz_lines[1][7] if len(mrz_lines[1]) > 7 else ""
            except:
                mrz_code_dict["sex"] = ""

            try:
                mrz_code_dict["expiry_date"] = mrz_lines[1][8:14]
                if len(mrz_lines[1]) > 14:
                    expiry_date_checkdigit = self._get_checkdigit(mrz_code_dict["expiry_date"])
                    if expiry_date_checkdigit != mrz_lines[1][14]:
                        checksum_failures.append("Expiry date checksum mismatch")
                    if include_checkdigit:
                        mrz_code_dict["expiry_date_checkdigit"] = expiry_date_checkdigit
                mrz_code_dict["expiry_date"] = self._format_date(mrz_code_dict["expiry_date"])
                
                # Calculate birth date with expiry date context
                if mrz_code_dict["birth_date"] and mrz_code_dict["expiry_date"]:
                    mrz_code_dict["birth_date"] = self._get_birth_date(mrz_code_dict["birth_date"], mrz_code_dict["expiry_date"])
            except:
                mrz_code_dict["expiry_date"] = ""

            try:
                mrz_code_dict["nationality_code"] = mrz_lines[1][15:18]
                if not mrz_code_dict["nationality_code"].isalpha():
                    mrz_code_dict["nationality_code"] = ""
            except:
                mrz_code_dict["nationality_code"] = ""

            try:
                mrz_code_dict["optional_data_2"] = mrz_lines[1][18:29].strip("<")
            except:
                mrz_code_dict["optional_data_2"] = ""
            
            # Final checkdigit (non-critical)
            try:
                final_checkdigit = self._get_final_checkdigit(mrz_lines, mrz_code_dict["mrz_type"])
                if mrz_lines[1][-1] != final_checkdigit:
                    checksum_failures.append("Final checksum mismatch")
                if include_checkdigit:
                    mrz_code_dict["final_checkdigit"] = final_checkdigit
            except:
                pass

            # Line 3 - Names
            try:
                if len(mrz_lines) > 2:
                    names = mrz_lines[2].split("<<")
                    mrz_code_dict["surname"] = names[0].replace("<", " ").strip() if len(names) > 0 else ""
                    mrz_code_dict["given_name"] = names[1].replace("<", " ").strip() if len(names) > 1 else ""
                else:
                    mrz_code_dict["surname"] = ""
                    mrz_code_dict["given_name"] = ""
            except:
                mrz_code_dict["surname"] = ""
                mrz_code_dict["given_name"] = ""

        # Add issue_date as empty since it's not in MRZ but needed in relevant_fields
        mrz_code_dict["issue_date"] = ""

        mrz_code_dict["mrz_text"] = mrz_text

        # Determine final status based on relevant fields extraction
        extracted_relevant_count = 0
        for field in relevant_fields:
            if mrz_code_dict.get(field) and str(mrz_code_dict[field]).strip():
                extracted_relevant_count += 1

        extraction_rate = extracted_relevant_count / len(relevant_fields)
        
        # Only mark as FAILURE if we couldn't extract any relevant fields
        if extraction_rate == 0:
            mrz_code_dict["status"] = "FAILURE"
            mrz_code_dict["status_message"] = "No relevant data could be extracted from MRZ"
        else:
            # Success with optional warnings about checksum issues
            mrz_code_dict["status"] = "SUCCESS"
            if checksum_failures:
                mrz_code_dict["status_message"] = f"Extracted {extracted_relevant_count}/{len(relevant_fields)} fields. Warnings: {'; '.join(checksum_failures)}"
            else:
                mrz_code_dict["status_message"] = f"Successfully extracted {extracted_relevant_count}/{len(relevant_fields)} relevant fields"
        
        # Add extraction metadata
        mrz_code_dict["extraction_rate"] = extraction_rate
        mrz_code_dict["extracted_relevant_count"] = extracted_relevant_count
        mrz_code_dict["total_relevant_fields"] = len(relevant_fields)
        mrz_code_dict["checksum_failures"] = checksum_failures

        return mrz_code_dict

    def validate_mrz(self, mrz_text):
        mrz_text = self._cleanse_roi(mrz_text)

        result = self._parse_mrz(mrz_text)
        if result.get("status") == "SUCCESS":
            return {"is_valid": True, "status_message": "The given mrz is valid"}
        else:
            return {"is_valid": False, "status_message": result.get("status_message")}

    def get_details(self, input_data, input_type="imagepath", ignore_parse=False, include_checkdigit=True):
        if input_type == "imagepath":
            if not self._is_valid(input_data):
                raise ValueError("Input is not a valid image file.")
            base64_string = self._image_to_base64(input_data)
            image_array = self._base64_to_array(base64_string)
            mrz_text = self._get_mrz(image_array)

            return mrz_text if ignore_parse else self._parse_mrz(mrz_text, include_checkdigit=include_checkdigit)
        elif input_type == "numpy":
            if not self._is_valid(input_data):
                raise ValueError("Input is not a valid NumPy array.")
            mrz_text = self._get_mrz(input_data)

            return mrz_text if ignore_parse else self._parse_mrz(mrz_text, include_checkdigit=include_checkdigit)
        elif input_type == "base64":
            image_array = self._base64_to_array(input_data)
            mrz_text = self._get_mrz(image_array)

            return mrz_text if ignore_parse else self._parse_mrz(mrz_text, include_checkdigit=include_checkdigit)
        elif input_type == "pdf":
            # get_details_from_pdf(input_data, ignore_parse=False, include_checkdigit=True)
            pass
        elif input_type == "text":
            mrz_text = self._cleanse_roi(input_data)

            return mrz_text if ignore_parse else self._parse_mrz(mrz_text, include_checkdigit=include_checkdigit)
        else:
            raise ValueError(f"Unsupported input_type: {input_type}")
