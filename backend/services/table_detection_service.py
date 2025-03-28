import time
import os
import json
import cv2
# import re
# import torch
import numpy as np
import pandas as pd
import tensorflow as tf
# import supervision as sv
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# from matplotlib.patches import Patch
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
from paddleocr import PaddleOCR, draw_ocr
from pdf2image import convert_from_path
from google import genai
from google.genai import types
from unidecode import unidecode
from fuzzywuzzy import fuzz
from datetime import datetime
# import base64
# import tempfile

from core.config import MODEL_SIGNATURE_PATH, MODEL_TABLE_TITLE_PATH, DEVICE, POPPLER_PATH, financial_tables, financial_tables_general,  models, EXTRACTED_FOLDER
from utils import retry_api_call, json_to_dataframe, json_to_dataframe_table, json_to_dataframe_title
from secret import api_keys


class TableDetectService:
    def __init__(self):
        print("TableDetectService: Khởi tạo là load model...")
        self.ocr = PaddleOCR(lang="en")
        self.table_title_detection_model = YOLO(MODEL_TABLE_TITLE_PATH).to(DEVICE)
        self.signature_detection_model = YOLO(MODEL_SIGNATURE_PATH).to(DEVICE)
        self.detection_class_names = ["table", "table rotated"]
        self.structure_class_map = {
            k: v
            for v, k in enumerate(
                [
                    "table",
                    "table column",
                    "table row",
                    "table column header",
                    "table projected row header",
                    "table spanning cell",
                    "no object",
                ]
            )
        }
        self.structure_class_thresholds = {
            "table": 0.5,
            "table column": 0.5,
            "table row": 0.5,
            "table column header": 0.5,
            "table projected row header": 0.5,
            "table spanning cell": 0.5,
            "no object": 10,  # Giá trị cao để loại bỏ "no object"
        }

    def detect_table(self, pdf_path, file_name_origin):
        """
        Flow xử lý file pdf như sau:
        - Chuyển file pdf thành danh sách các hình ảnh
        - Lặp qua từng hình và xử lý như sau:
            + Nếu hình đó có chứa table:
                > Sử dụng model best_model_YOlO để detect ra bẳng
        """
        recognized_titles_set = set() # biến này để tạo 1 set lưu trữ những bảng mình đã trích xuất, khi nào đủ 3 bảng rồi thì dừng chương tình
        dfs_dict = {} # Biến này để concat dữ liệu từng bảng, vì bảng có thể dài quá bị chuyển sang page khác


        images = self.pdf_to_images(pdf_path)  # Chuyển pdf thành hình ảnh

        index_start = 0  # Bắt đầu từ ảnh đầu tiên
        while index_start < len(images):
            index_chuky = None  # Reset mỗi lần lặp 
            for i in range(index_start, len(images)):
                selected_images = [] #Tạo 1 list để lưu các bảng chung 1 title đi từ title đến ảnh chữ ký đầu tiên nhận diện được
                image = images[i]
                print(f"🔍 Đang xử lý ảnh {i+1}")

                # Nhận diện bảng -> table-title
                nhandien_table = self.table_detection(image)

                if not nhandien_table:
                    continue  # Nếu không có bảng, bỏ qua ảnh này

                has_rotated_table = any(
                    self.detection_class_names[det[5]] == "table rotated"
                    for det in nhandien_table
                )
                
                # Chỉ xoay ảnh nếu có bảng xoay
                image_to_process = (
                    self.table_rotation(image, nhandien_table) if has_rotated_table else image
                )

                # Nhận diện tiêu đề --> phân vân
                # df_title, text_title = self.detect_and_extract_title(
                #     image_to_process,
                #     "/content/drive/MyDrive/Test AI ở Orient/AI_for_Finance/Bản sao của best_model_YoLo.pt",
                #     ocr,
                # )

                df_title, text_title = self.detect_and_extract_title(image_to_process) # hàm nhận diện các text ngoài bảng và trả về 1 DataFrame và 1 biến lưu text để cho LLM nhận ngữ cảnh

                # Để sleep để giúp model nghỉ, bị limit 1 phút không quá 2 lần
                time.sleep(45)

                for api_key in api_keys:
                  json_title = retry_api_call(generate_title, model, api_keys[api_key]['title'], dataframe_to_json(df_title), text_title)
                  if json_title:
                      break
                print("Hoàn tất thử API.")

                data_title = json_to_dataframe_title(json_title)  # Kết quả title của bảng
                recognized_title = self.recognize_financial_table(
                    data_title, financial_tables, threshold=80
                )  # Nhận diện xem title của bảng là gì có phù hợp với 3 tên bảng dự án đề ra không

                # Nếu nhận diện được title, thêm vào danh sách nhận diện
                if not (recognized_title):
                    continue
                # Tìm ảnh chữ ký tiếp theo sau ảnh title
                for j in range(images.index(image), len(images)):
                    nhandien_chuky = images[j]
                    results_chuky = self.detect_signature(nhandien_chuky)
                    if results_chuky[0]:
                        index_chuky = j  # Lưu vị trí ảnh chữ ký
                        print(f"🖊 Ảnh chữ ký được phát hiện ở {index_chuky +1 }")
                        break

                # Lấy danh sách ảnh từ title đến chữ ký
                if index_chuky:
                    selected_images.extend(images[images.index(image) : index_chuky + 1])

                # Vòng lặp qua ảnh từ title đến chữ ký để trích xuất bảng
                if selected_images:
                  pre_name_column = None
                  for img in selected_images:
                      processed_image = Process_Image(img)
                      # 2️⃣ Chuyển đổi ảnh sang CMYK và lấy kênh K
                      _, _, _, black_channel = rgb_to_cmyk(processed_image)
                      # 3️⃣ Điều chỉnh độ sáng & độ tương phản
                      processed_image = adjust_contrast(black_channel, alpha=2.0, beta=-50)
                      if processed_image is not None:
                          df_table, text_table = process_pdf_image(processed_image, ocr)
                          if not df_table.empty:
                                if (len(df_table) < 101) and (len(df_table.columns) < 10):
                                  token = 9000
                                elif (len(df_table) < 201) and (len(df_table.columns) < 10):
                                  token = 18000
                                else:
                                  token = 30000
                                time.sleep(45)
                                if selected_images.index(img) ==0:
                                  response_schema=generate_json_schema(dataframe_to_json(df_table))
                                for api_key in api_keys:
                                  json_table = retry_api_call(generate_table, model, api_keys[api_key]['table'],  dataframe_to_json(df_table), text_table, token, pre_name_column, response_schema)
                                  if json_table:
                                    break
                                print("Hoàn tất thử API.")

                                data_table = json_to_dataframe_table(json_table)

                                if selected_images.index(img) ==0:
                                  found = False  # Flag để thoát cả hai vòng lặp khi tìm thấy kết quả
                                  recognized_title = "Bảng cân đối kế toán"
                                  for column in data_table.columns:
                                      for value in data_table[column].dropna():
                                          value = normalize_text(value)
                                          print(value)
    
                                          if "luu chuyen" in value:
                                              recognized_title = "Báo cáo lưu chuyển tiền tệ"
                                              found = True
                                              break  # Thoát khỏi vòng lặp giá trị trong cột
    
                                          elif "doanh thu ban hang" in value or "ban hang" in value:
                                              recognized_title = "Báo cáo KQHĐKD"
                                              found = True
                                              break  # Thoát khỏi vòng lặp giá trị trong cột
                                      if found:
                                          break  # Thoát khỏi vòng lặp cột
                                print(f"Fix nhận diện được là {recognized_title}")

                                recognized_titles_set.add(recognized_title) #Lưu nó vào set đã tạo ở trước, giúp nhận diện nào đủ 3 bảng thì dừng lại
                                # display(data_table)
                                if selected_images.index(img) == 0:
                                    pre_name_column = data_table.columns.tolist() # Vì model LLM hay trả kết quả về 1 lúc 1 khác nên dùng tên các cột của bảng đầu tiên làm chuẩn để các bảng sau trả về cho chuẩn
                                else:
                                    if len(data_table.columns) == len(pre_name_column):
                                        data_table.columns = pre_name_column
                                    else:
                                        data_table = data_table.reindex(
                                            columns=pre_name_column, fill_value=None
                                        )

                                if not data_table.empty:                            #Ở đây kiểm tra trong biến dfs_dict đã có DataFrame với key là tiitle đang nhận diện chưa, nếu chưa thì lưu vào biến dfs_dict DataFrame với key đấy
                                    if recognized_title not in dfs_dict:            # Nếu ở trong dfs_dict đã có DataFrame với key là titile đang nhận diện thì nó sẽ nối (concat dữ liệu lại dựa trên key là title đấy)
                                        dfs_dict[recognized_title] = data_table
                                    else:
                                        dfs_dict[recognized_title] = pd.concat(
                                            [dfs_dict[recognized_title], data_table],
                                            ignore_index=True,
                                        )
                    # display(dfs_dict[recognized_title])

                    break # beak để cập nhật lại ví trí bắt đầu là 
                
            # Cập nhật vị trí bắt đầu cho vòng lặp tiếp theo. Vì cách chạy là ví dụ bảng Cân đối kế toán ở trước
            # Thì xác định được index_chuky của bảng Cân đối kế toán rồi thì update lại lên vòng lặp cho nó chạy từ chữ ký chạy tiếp.
            if index_chuky:        
                index_start = index_chuky + 1
            else:
                index_start = i + 1
                # Kiểm tra nếu đã nhận diện đủ bảng tài chính thì dừng
            if recognized_titles_set == set(financial_tables):
                print("✅ Đã nhận diện đủ tất cả bảng tài chính. Dừng lại!")
                break

        # Lưu kết quả vào file Excel
        name, _ = file_name_origin.rsplit(".", 1) if "." in file_name_origin else (file_name_origin, "")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # Định dạng thời gian: YYYYMMDD_HHMMSS
        new_name = f"{name}_{timestamp}.xlsx"
        file_path = os.path.join(EXTRACTED_FOLDER, new_name)
        with pd.ExcelWriter(file_path, engine="xlsxwriter") as writer: # TODO: No module named 'xlsxwriter'
            for i, (sheet_name, df) in enumerate(dfs_dict.items()):
                df.to_excel(writer, sheet_name=sheet_name[:31], index=False)

                # Nếu không phải lần cuối cùng, thì chờ trước khi gửi request tiếp theo
                if i < len(dfs_dict) - 1:
                    print(f"Chờ 30 giây trước khi tiếp tục lưu bảng tiếp theo...")
                    time.sleep(30)  # Chờ 30 giây giữa các request

        print(f"File Excel đã được lưu tại: {file_path}")
        download_url = f"/{EXTRACTED_FOLDER}/{new_name}"

        return dfs_dict, download_url
    
        
    def pdf_to_images(self, pdf_path):
        images = convert_from_path(pdf_path, poppler_path=POPPLER_PATH)
        return images

    # model table_title_detection_model
    def table_detection(self, image):
        imgsz = 800
        pred = self.table_title_detection_model.predict(image, imgsz=imgsz)
        pred = pred[0].boxes
        result = pred.cpu().numpy()
        result_list = [
            list(result.xywhn[i]) + [result.conf[i], int(result.cls[i])]
            for i in range(result.shape[0])
        ]
        return result_list

    def table_rotation(self, image, list_detection_table):
        for det in list_detection_table:
            x_center, y_center, w_n, h_n, conf, cls_id = det
            if self.detection_class_names[cls_id] == "table rotated":
                print("This is a rotated table")
                image = image.rotate(-90, expand=True)
            img = image.convert("L")
            thresh_img = img.point(lambda p: 255 if p > 120 else 0)
            return thresh_img

    # model table_title_detection_model
    def Process_Image(self, image):
        results = self.table_title_detection_model.predict(image, task="detect")
        boxes = results[0].boxes

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls_id = int(box.cls[0])
            class_name = self.table_title_detection_model.names[cls_id]

            # Chuyển đổi sang PIL nếu cần
            if isinstance(image, np.ndarray):
                image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            # Cắt ảnh trước
            cropped_table = image.crop((int(x1), int(y1), int(x2), int(y2)))

            # plt.imshow(cropped_table)
            # plt.title("Cropped Image")
            # plt.show()

            # Nếu là bảng bị xoay, xoay lại
            if class_name == "table rotated":
                print("This is a rotated table")
                cropped_img = cropped_table.rotate(-90, expand=True)
                return cropped_img  # Trả về ảnh đã cắt và sửa góc

            return cropped_table  # Nếu không bị xoay, trả về ảnh cắt nguyên bản

    # sử dụng model từ models = ["gemini-2.0-pro-exp-02-05", "gemini-2.0-flash-thinking-exp-01-21"]
    # chuyển API key sang file config
    def generate_title(self, model, API, temperature, top_p, top_k, path_title_json, text_title):
        result = ""
        client = genai.Client(api_key=f"{API}")

        # Mở file JSON và đọc nội dung
        file_path = path_title_json
        with open(file_path, "r", encoding="utf-8") as f:
            json_content = json.load(f)  # Load JSON thành dict

        model = model
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(
                        text=f"""Mình đang trích xuất dữ liệu từ hình ảnh chứa bảng tài chính bằng PaddleOCR. Dữ liệu nhận diện được lưu trong {text_title}.
    Tuy nhiên, dữ liệu gặp lỗi:
    - Sai chính tả tiếng Việt trong báo cáo tài chính, kế toán và dòng tiền
    - Lỗi ngữ pháp tiếng Việt trong báo cáo tài chính, kế toán và dòng tiền
    Vì đây là một báo cáo quan trọng, rất nhiều thứ ảnh hưởng xấu đến nếu như nó sai chính tả và lỗi ngữ pháp.
    Bạn hãy trả về cho mình một DataFrame chỉ có 1 cột là "values" chứa các giá trị được ngăn cách thành từng dòng giúp người đọc dễ dàng đọc hiêu, mỗi hàng không chứa lồng ghép thành chuỗi hay danh sách gì, chỉ 1 dòng là 1 giá trị riêng biệt từ file JSON gốc.
                        Dữ liệu JSON gốc:
                        {json.dumps(json_content, indent=2, ensure_ascii=False)}
                        """
                    ),
                ],
            ),
        ]

        generate_content_config = types.GenerateContentConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_output_tokens=8192,
            response_mime_type="application/json",
        )

        for chunk in client.models.generate_content_stream(
            model=model,
            contents=contents,
            config=generate_content_config,
        ):
            result += chunk.text
        return result

    # sử dụng model từ models = ["gemini-2.0-pro-exp-02-05", "gemini-2.0-flash-thinking-exp-01-21"]
    # chuyển API key sang file config
    def generate_table(self, model, API, temperature, top_p, top_k, path_dataframe_json, text_table, token, table_columns):
        result = ""
        client = genai.Client(api_key=f"{API}")  # Đuôi nc

        # Mở file JSON và đọc nội dung
        file_path = path_dataframe_json
        with open(file_path, "r", encoding="utf-8") as f:
            json_content = json.load(f)  # Load JSON thành dict

        model = model
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(
                        text=f"""Mình đang trích xuất dữ liệu từ hình ảnh chứa bảng tài chính bằng PaddleOCR. Dữ liệu nhận diện được lưu trong {text_table}.
    Tuy nhiên, dữ liệu gặp lỗi:
    - Sai chính tả tiếng Việt
    - Lỗi ngữ pháp tiếng Việt
    - Sắp xếp sai dòng/cột, ảnh hưởng đến tính chính xác của báo cáo tài chính.

    Bạn hãy giúp mình chuẩn hóa lại bảng dữ liệu dựa vào bối cảnh và ký tự nhận diện được trong {text_table} và kiến thức chuyên ngành tài chính, kế toán, đảm bảo đúng thuật ngữ, chính tả và cấu trúc bảng hợp lý (gồm dòng, cột, tiêu đề cột, dữ liệu trong bảng). Kết quả trả về là một DataFrame chuẩn theo đúng định dạng bảng báo cáo tài chính cho người dùng dễ dàng đọc hiểu,
    đảm bảo đúng thông tin được truyền vào từ biến {text_table} và Dữ liệu JSON gốc không sai kết quả.
    Đây là báo cáo kết quả hoạt động kinh doanh của công ty ABC.
    Bạn hãy kiểm tra nếu danh sách tên cột {table_columns} rỗng thì hãy nhận diện để đặt tên cột mặc định bắt buộc phải có chứa 3 cột: "Mã số", "Tên chỉ tiêu", "Thuyết minh" và chuẩn hóa các cột sau: "Mã số", "Tên chỉ tiêu", "Thuyết minh".
    Nếu danh sách tên cột {table_columns} không rỗng thì hãy đặt tên cột giống như từng giá trị trong {table_columns} và chuẩn hóa chúng đúng với kiến thức quan trọng cần thiết trong báo cáo tài chính.
    Tự động nhận diện và chuẩn hóa các cột số liệu, đảm bảo chúng được hiển thị đúng định dạng (ví dụ: số nguyên, số thập phân, đơn vị tiền tệ).
    Nếu có thể, hãy xác định năm tài chính được đề cập trong báo cáo và sử dụng thông tin này để đặt tên cho các cột số liệu (ví dụ: "Năm 2022", "Năm 2023").
    Sử dụng tên cột có dấu cách và viết hoa chữ cái đầu tiên của mỗi từ.

    Dữ liệu JSON gốc:
    {json.dumps(json_content, indent=2, ensure_ascii=False)}
    """
                    ),
                ],
            ),
        ]

        generate_content_config = types.GenerateContentConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_output_tokens=token,
            response_mime_type="application/json",
        )

        for chunk in client.models.generate_content_stream(
            model=model,
            contents=contents,
            config=generate_content_config,
        ):
            result += chunk.text
        return result

    def process_image_ocr(self, image):
        """Nhận diện text trong ảnh bằng OCR."""
        if isinstance(image, Image.Image):
            image = np.array(image)
        output = self.ocr.ocr(image)[0]
        boxes = [line[0] for line in output]
        texts = [line[1][0] for line in output]
        probabilities = [line[1][1] for line in output]
        return image, boxes, texts, probabilities

    def get_horizontal_vertical_boxes(self, image, boxes):
        """Tạo danh sách các bounding box ngang và dọc."""
        image_height, image_width = image.shape[:2]
        horiz_boxes = []
        vert_boxes = []

        for box in boxes:
            x_h, x_v = 0, int(box[0][0])
            y_h, y_v = int(box[0][1]), 0
            width_h, width_v = image_width, int(box[2][0] - box[0][0])
            height_h, height_v = int(box[2][1] - box[0][1]), image_height

            horiz_boxes.append([x_h, y_h, x_h + width_h, y_h + height_h])
            vert_boxes.append([x_v, y_v, x_v + width_v, y_v + height_v])

        return horiz_boxes, vert_boxes

    def apply_non_max_suppression(self, boxes, scores, image):
        """Áp dụng Non-Max Suppression (NMS) để loại bỏ các bounding box dư thừa."""
        nms_indices = tf.image.non_max_suppression(
            boxes,
            scores,
            max_output_size=1000,
            iou_threshold=0.1,
            score_threshold=float("-inf"),
        ).numpy()
        return np.sort(nms_indices)

    def intersection(self, box_1, box_2):
        """Tính toán giao giữa hai bbox."""
        return [box_2[0], box_1[1], box_2[2], box_1[3]]

    def iou(self, box_1, box_2):
        """Tính chỉ số Intersection over Union (IoU)."""
        x_1, y_1 = max(box_1[0], box_2[0]), max(box_1[1], box_2[1])
        x_2, y_2 = min(box_1[2], box_2[2]), min(box_1[3], box_2[3])
        inter = abs(max((x_2 - x_1, 0)) * max((y_2 - y_1), 0))

        if inter == 0:
            return 0
        box_1_area = abs((box_1[2] - box_1[0]) * (box_1[3] - box_1[1]))
        box_2_area = abs((box_2[2] - box_2[0]) * (box_2[3] - box_2[1]))

        return inter / float(box_1_area + box_2_area - inter)

    def extract_table_data(self, boxes, texts, horiz_lines, vert_lines, horiz_boxes, vert_boxes):
        """Trích xuất dữ liệu bảng từ bbox đã nhận diện."""
        out_array = [["" for _ in range(len(vert_lines))] for _ in range(len(horiz_lines))]

        unordered_boxes = [vert_boxes[i][0] for i in vert_lines]
        ordered_boxes = np.argsort(unordered_boxes)

        for i in range(len(horiz_lines)):
            for j in range(len(vert_lines)):
                resultant = self.intersection(
                    horiz_boxes[horiz_lines[i]], vert_boxes[vert_lines[ordered_boxes[j]]]
                )
                for b, box in enumerate(boxes):
                    the_box = [box[0][0], box[0][1], box[2][0], box[2][1]]
                    if self.iou(resultant, the_box) > 0.1:
                        out_array[i][j] = texts[b]

        return pd.DataFrame(np.array(out_array))

    def process_pdf_image(self, image):
        """Hàm tổng hợp để xử lý ảnh từ PDF, nhận diện bảng và trích xuất dữ liệu."""
        # OCR trích xuất text & bbox
        image, boxes, texts, probabilities = self.process_image_ocr(image)

        # Nhận diện box ngang & dọc
        horiz_boxes, vert_boxes = self.get_horizontal_vertical_boxes(image, boxes)

        # Loại bỏ các box dư thừa bằng Non-Max Suppression
        horiz_lines = self.apply_non_max_suppression(horiz_boxes, probabilities, image)
        vert_lines = self.apply_non_max_suppression(vert_boxes, probabilities, image)

        # Trích xuất dữ liệu bảng thành DataFrame
        df = self.extract_table_data(
            boxes, texts, horiz_lines, vert_lines, horiz_boxes, vert_boxes
        )

        return df, texts

    # model nhận diện chữ kí
    def detect_signature(self, image):
        return self.signature_detection_model(image)

    # model nhận diện table title
    def detect_and_extract_title(self, image):

        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype(np.uint8))

        
        # Nhận diện table trong ảnh để cắt phần title
        results = self.table_title_detection_model(image)

        # Lấy ảnh gốc
        img_last = results[0].orig_img.copy()

        # Lấy danh sách tọa độ chữ ký (x1, y1, x2, y2)
        boxes_obj = results[0].boxes
        if boxes_obj is not None and len(boxes_obj) > 0:
            coords = boxes_obj.xyxy.cpu().numpy()  # Chuyển về numpy array
            x1, y1, x2, y2 = map(int, coords[0])  # Lấy tọa độ đầu tiên (nếu có nhiều)

            # Lấy kích thước ảnh
            h, w, _ = img_last.shape

            # Cắt vùng trên và dưới của chữ ký
            top_region = img_last[0:y1, 0:w]
            bottom_region = img_last[y2:h, x1:x2]

            # Nhận diện văn bản từ hai vùng
            top_text = self.ocr.ocr(top_region)[0]
            bottom_text = self.ocr.ocr(bottom_region)[0]

            # Lọc kết quả nhận diện
            top_result = [
                line[1][0]
                for line in (top_text or [])  # Nếu None thì chuyển thành list rỗng
                if line and len(line) > 1 and line[1] and len(line[1]) > 0
            ]

            bottom_result = [
                line[1][0]
                for line in (bottom_text or [])  # Nếu None thì chuyển thành list rỗng
                if line and len(line) > 1 and line[1] and len(line[1]) > 0
            ]

            # Gộp kết quả từ cả hai vùng
            extracted_text = top_result + bottom_result
        else:
            extracted_text = []
        df_title = pd.DataFrame(extracted_text)
        return df_title, extracted_text

    def normalize_text(self, text):
        return unidecode(str(text)).lower().strip()

    def recognize_financial_table(self, df, financial_tables, threshold=80):
        """
        Nhận diện tiêu đề bảng tài chính từ một DataFrame.

        Args:
            df (pd.DataFrame): DataFrame chứa dữ liệu cần kiểm tra.
            financial_tables (list): Danh sách các bảng tài chính chuẩn.
            image : Ảnh đang xét
            threshold (int): Ngưỡng độ tương đồng tối thiểu để chấp nhận.
        Returns:
            tuple: (Tên bảng tài chính nhận diện được, ảnh tương ứng)
        """
        # Chuẩn hóa danh sách bảng tài chính
        normalized_tables = [self.normalize_text(table) for table in financial_tables]

        # Duyệt qua từng cột trong DataFrame
        for column in df.columns:
            for value in df[column].dropna():  # Bỏ qua giá trị NaN
                norm_value = self.normalize_text(value)

                # Kiểm tra khớp chính xác trước
                if norm_value in normalized_tables:
                    print(f"✅ Khớp chính xác: {value} (cột: {column})")
                    recognized_title = financial_tables[normalized_tables.index(norm_value)]
                    return recognized_title

                # Nếu không khớp chính xác, kiểm tra độ tương đồng
                for norm_table in normalized_tables:
                    similarity = fuzz.partial_ratio(norm_value, norm_table)
                    if similarity >= threshold:
                        print(
                            f"🔹 Khớp tương đồng ({similarity}%): {value} ~ {norm_table} (cột: {column})"
                        )
                        recognized_title = financial_tables[
                            normalized_tables.index(norm_table)
                        ]
                        return recognized_title

        print("❌ Không tìm thấy bảng tài chính nào phù hợp.")
        return None

    def get_model_params(self, model):
        if model == "gemini-2.0-pro-exp-02-05":
            return 1, 0.95, 64
        elif model == "gemini-2.0-flash-thinking-exp-01-21":
            return 0.7, 0.95, 64
        return None
