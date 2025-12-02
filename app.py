from flask import Flask, render_template, request, send_file, jsonify
import os
import tempfile
import traceback
import uuid
import json
import fitz  # PyMuPDF
import easyocr
import pandas as pd
import numpy as np
import cv2
from werkzeug.utils import secure_filename
from collections import defaultdict

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # Максимальный размер файла 50MB
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()

# Включаем логирование ошибок
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Разрешенные расширения
ALLOWED_EXTENSIONS = {'pdf'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Lazy-инициализация EasyOCR, чтобы Render не ждал долгий старт при импорте модуля
reader = None

def get_reader():
    global reader
    if reader is None:
        logger.info("Инициализация EasyOCR reader (lazy)...")
        # можно отключить прогрессбар, чтобы не засорять логи Render
        reader = easyocr.Reader(['en', 'ru'], verbose=False)
        logger.info("EasyOCR reader инициализирован.")
    return reader

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/favicon.ico')
def favicon():
    return '', 204  # No content

@app.route('/upload', methods=['POST'])
def upload_file():
    pdf_paths = []
    json_path = None
    
    try:
        if 'files[]' not in request.files and 'file' not in request.files:
            return jsonify({'error': 'Файлы не найдены'}), 400
        
        # Получаем список файлов
        files = request.files.getlist('files[]') or request.files.getlist('file')
        
        if not files or all(f.filename == '' for f in files):
            return jsonify({'error': 'Файлы не выбраны'}), 400
        
        # Фильтруем валидные PDF файлы
        valid_files = []
        for file in files:
            if file.filename and allowed_file(file.filename):
                valid_files.append(file)
        
        if not valid_files:
            return jsonify({'error': 'Не найдено валидных PDF файлов'}), 400
        
        # Сохраняем все файлы
        for file in valid_files:
            filename = secure_filename(file.filename)
            pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{uuid.uuid4().hex[:8]}_{filename}")
            file.save(pdf_path)
            pdf_paths.append(pdf_path)
            logger.info(f"Файл сохранен: {pdf_path}")
        
        # Обрабатываем все PDF файлы и объединяем в один JSON
        json_path = process_multiple_pdfs_to_json(pdf_paths)
        logger.info(f"Объединенный JSON файл создан: {json_path}")
        
        # Отправляем JSON с ссылкой на файл
        return jsonify({
            'json_url': f'/download/json/{os.path.basename(json_path)}',
            'json_filename': os.path.basename(json_path),
            'files_processed': len(valid_files)
        })
    
    except Exception as e:
        error_msg = str(e)
        error_trace = traceback.format_exc()
        logger.error(f"Ошибка при обработке файлов: {error_msg}\n{error_trace}")
        return jsonify({'error': f'Ошибка при обработке файлов: {error_msg}'}), 500
    finally:
        # Удаляем временные PDF файлы
        for pdf_path in pdf_paths:
            if pdf_path and os.path.exists(pdf_path):
                try:
                    os.remove(pdf_path)
                    logger.info(f"Временный PDF файл удален: {pdf_path}")
                except Exception as e:
                    logger.warning(f"Не удалось удалить временный файл {pdf_path}: {e}")
        
        # Удаляем JSON файл через задержку
        if json_path and os.path.exists(json_path):
            import threading
            def delayed_delete(path, delay=3600):
                import time
                time.sleep(delay)
                try:
                    if os.path.exists(path):
                        os.remove(path)
                        logger.info(f"Временный файл удален: {path}")
                except Exception as e:
                    logger.warning(f"Не удалось удалить временный файл {path}: {e}")
            threading.Thread(target=delayed_delete, args=(json_path,), daemon=True).start()

def process_multiple_pdfs_to_json(pdf_paths):
    """Обрабатывает несколько PDF файлов и объединяет результаты в один JSON"""
    all_pages = []
    all_metadata = {
        'total_files': len(pdf_paths),
        'total_pages': 0,
        'total_text_blocks': 0,
        'files': [],
        'average_confidence': 0.0,
        'description': 'Объединенный OCR результат нескольких PDF файлов. Используйте structured_table для анализа данных.'
    }
    
    confidence_scores = []
    
    for file_idx, pdf_path in enumerate(pdf_paths, 1):
        filename = os.path.basename(pdf_path)
        logger.info(f"Обработка файла {file_idx}/{len(pdf_paths)}: {filename}")
        
        # Обрабатываем один PDF
        df, page_dimensions = process_pdf_to_dataframe(pdf_path)
        
        if df is None or df.empty:
            logger.warning(f"Файл {filename} не содержит данных")
            continue
        
        # Создаем структурированные таблицы для этого файла
        structured_tables = create_structured_tables(df)
        
        # Получаем метаданные файла
        file_confidence = float(df['confidence'].mean())
        confidence_scores.append(file_confidence)
        
        file_info = {
            'file_index': file_idx,
            'filename': filename,
            'pages_count': len(df['page'].unique()),
            'text_blocks_count': len(df),
            'average_confidence': file_confidence
        }
        all_metadata['files'].append(file_info)
        all_metadata['total_pages'] += len(df['page'].unique())
        all_metadata['total_text_blocks'] += len(df)
        
        # Обрабатываем каждую страницу
        for page_num in sorted(df['page'].unique()):
            page_data = df[df['page'] == page_num]
            
            # Определяем тип документа
            all_text = ' '.join(page_data['text'].astype(str)).lower()
            doc_type = 'unknown'
            if 'оборотно-сальдовая' in all_text or 'оборотная' in all_text:
                doc_type = 'trial_balance'
            elif 'баланс' in all_text:
                doc_type = 'balance_sheet'
            elif 'отчет' in all_text:
                doc_type = 'report'
            
            page_info = {
                'file_index': file_idx,
                'filename': filename,
                'page_number': int(page_num),
                'document_type': doc_type,
                'text_blocks': page_data[['x0', 'y0', 'x1', 'y1', 'text', 'confidence']].to_dict('records'),
            }
            
            # Добавляем структурированную таблицу
            if page_num in structured_tables and structured_tables[page_num] is not None:
                table_df = structured_tables[page_num]
                
                structured_data = {
                    'columns': table_df.columns.tolist(),
                    'rows': [],
                    'row_count': len(table_df),
                    'column_count': len(table_df.columns),
                    'data_format': 'table',
                    'description': 'Структурированная таблица данных. Каждая строка в rows - это массив значений ячеек в порядке колонок.'
                }
                
                for idx, row in table_df.iterrows():
                    row_dict = {}
                    for col in table_df.columns:
                        val = row[col]
                        # Проверяем на NaN правильно - используем at для получения скалярного значения
                        try:
                            scalar_val = table_df.at[idx, col]
                            if pd.isna(scalar_val):
                                row_dict[col] = ''
                            else:
                                row_dict[col] = str(scalar_val)
                        except:
                            # Fallback если at не работает
                            val_str = str(val) if val is not None and str(val) != 'nan' else ''
                            row_dict[col] = val_str
                    
                    values = []
                    for col in table_df.columns:
                        try:
                            scalar_val = table_df.at[idx, col]
                            if pd.isna(scalar_val):
                                values.append('')
                            else:
                                values.append(str(scalar_val))
                        except:
                            val_str = str(row[col]) if row[col] is not None and str(row[col]) != 'nan' else ''
                            values.append(val_str)
                    
                    structured_data['rows'].append({
                        'row_index': int(idx),
                        'cells': row_dict,
                        'values': values
                    })
                
                page_info['structured_table'] = structured_data
                page_info['structured_table_array'] = {
                    'headers': table_df.columns.tolist(),
                    'data': table_df.values.tolist()
                }
            
            # Добавляем заголовки документа
            header_texts = []
            for _, row in page_data.iterrows():
                if row['y0'] < 400:
                    header_texts.append({
                        'text': row['text'],
                        'position': {'x0': float(row['x0']), 'y0': float(row['y0'])},
                        'confidence': float(row['confidence'])
                    })
            page_info['document_headers'] = header_texts
            
            all_pages.append(page_info)
    
    # Вычисляем среднюю уверенность
    if confidence_scores:
        all_metadata['average_confidence'] = float(sum(confidence_scores) / len(confidence_scores))
    
    # Создаем объединенный JSON
    result = {
        'metadata': all_metadata,
        'pages': all_pages
    }
    
    # Сохраняем JSON файл
    file_id = uuid.uuid4().hex[:8]
    json_filename = f'ocr_combined_{file_id}.json'
    json_path = os.path.join(app.config['UPLOAD_FOLDER'], json_filename)
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    return json_path

def process_pdf_to_dataframe(pdf_path):
    """Обрабатывает один PDF и возвращает DataFrame и размеры страниц"""
    all_data = []
    page_dimensions = {}
    
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF файл не найден: {pdf_path}")
    
    doc = fitz.open(pdf_path)
    
    try:
        for page_number in range(len(doc)):
            page = doc[page_number]
            pix = page.get_pixmap(dpi=300)
            
            page_dimensions[page_number + 1] = {
                'width': pix.width,
                'height': pix.height
            }
            
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            
            if pix.n == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            elif pix.n == 1:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            
            result = reader.readtext(img)
            
            for (bbox, text, prob) in result:
                x0, y0 = bbox[0]
                x1, y1 = bbox[2]
                all_data.append({
                    "page": page_number + 1,
                    "x0": x0,
                    "y0": y0,
                    "x1": x1,
                    "y1": y1,
                    "text": text,
                    "confidence": prob
                })
    finally:
        doc.close()
    
    if not all_data:
        return None, page_dimensions
    
    df = pd.DataFrame(all_data)
    return df, page_dimensions

def process_pdf_to_excel_and_html(pdf_path):
    """Обрабатывает PDF файл и возвращает пути к Excel, HTML, JSON и CSV файлам"""
    all_data = []
    page_dimensions = {}
    
    # Открываем PDF
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF файл не найден: {pdf_path}")
    
    doc = fitz.open(pdf_path)
    
    try:
        for page_number in range(len(doc)):
            page = doc[page_number]
            pix = page.get_pixmap(dpi=300)
            
            # Сохраняем размеры страницы
            page_dimensions[page_number + 1] = {
                'width': pix.width,
                'height': pix.height
            }
            
            # Конвертируем в numpy array для OpenCV
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            
            if pix.n == 4:  # RGBA -> BGR
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            elif pix.n == 1:  # Grayscale -> BGR
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            
            # Распознаем текст с координатами
            ocr_reader = get_reader()
            result = ocr_reader.readtext(img)
            # Распознаем текст с координатами
            ocr_reader = get_reader()
            result = ocr_reader.readtext(img)
            
            for (bbox, text, prob) in result:
                x0, y0 = bbox[0]
                x1, y1 = bbox[2]
                all_data.append({
                    "page": page_number + 1,
                    "x0": x0,
                    "y0": y0,
                    "x1": x1,
                    "y1": y1,
                    "text": text,
                    "confidence": prob
                })
    finally:
        doc.close()
    
    # Преобразуем в DataFrame
    if not all_data:
        raise ValueError("Не удалось распознать текст в PDF файле. Возможно, файл пуст или поврежден.")
    
    df = pd.DataFrame(all_data)
    
    # Создаем временные файлы с уникальным именем
    file_id = uuid.uuid4().hex[:8]
    excel_filename = f'parsed_table_easyocr_{file_id}.xlsx'
    excel_path = os.path.join(app.config['UPLOAD_FOLDER'], excel_filename)
    
    # Создаем структурированные таблицы
    structured_tables = create_structured_tables(df)
    
    # Сохраняем Excel с двумя листами: координаты и структурированные таблицы
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Координаты', index=False)
        for page_num, table_df in structured_tables.items():
            if table_df is not None and not table_df.empty:
                table_df.to_excel(writer, sheet_name=f'Таблица_Стр_{page_num}', index=False)
    
    # Создаем HTML визуализацию
    html_path = create_html_visualization(df, page_dimensions, file_id)
    
    # Создаем JSON файл со структурированными данными
    json_path = create_json_output(df, structured_tables, file_id)
    
    # Создаем CSV файл со структурированными данными
    csv_path = create_csv_output(structured_tables, file_id)
    
    return excel_path, html_path, json_path, csv_path

def create_structured_tables(df, y_tolerance=20, x_tolerance=30):
    """
    Группирует текст в структурированные таблицы на основе координат
    Улучшенный алгоритм с определением колонок через кластеризацию
    """
    structured = {}
    
    for page_num in sorted(df['page'].unique()):
        page_data = df[df['page'] == page_num].copy()
        if page_data.empty:
            continue
        
        # Вычисляем центры блоков текста
        page_data['center_x'] = (page_data['x0'] + page_data['x1']) / 2
        page_data['center_y'] = (page_data['y0'] + page_data['y1']) / 2
        page_data['width'] = page_data['x1'] - page_data['x0']
        page_data['height'] = page_data['y1'] - page_data['y0']
        
        # Фильтруем очень маленькие элементы (шум)
        min_area = 50
        page_data = page_data[
            (page_data['width'] * page_data['height']) >= min_area
        ].copy()
        
        if page_data.empty:
            continue
        
        # Группируем по строкам
        page_data = page_data.sort_values('center_y')
        rows = []
        current_row_y = None
        current_row = []
        
        for _, item in page_data.iterrows():
            y_center = item['center_y']
            if current_row_y is None or abs(y_center - current_row_y) <= y_tolerance:
                current_row.append(item)
                if current_row_y is None:
                    current_row_y = y_center
                else:
                    # Взвешенное среднее
                    current_row_y = (current_row_y * (len(current_row) - 1) + y_center) / len(current_row)
            else:
                if current_row:
                    rows.append(current_row)
                current_row = [item]
                current_row_y = y_center
        
        if current_row:
            rows.append(current_row)
        
        if not rows:
            structured[page_num] = None
            continue
        
        # Определяем позиции колонок через анализ всех X координат
        all_x_positions = []
        for row in rows:
            for item in row:
                all_x_positions.append(item['center_x'])
        
        if not all_x_positions:
            structured[page_num] = None
            continue
        
        # Кластеризуем X позиции для определения колонок
        all_x_positions = sorted(set(all_x_positions))
        column_positions = []
        if all_x_positions:
            column_positions.append(all_x_positions[0])
            for x in all_x_positions[1:]:
                # Если расстояние до ближайшей колонки больше порога - новая колонка
                min_dist = min(abs(x - col) for col in column_positions)
                if min_dist > x_tolerance:
                    column_positions.append(x)
        
        column_positions = sorted(column_positions)
        num_columns = len(column_positions)
        
        # Создаем таблицу
        table_rows = []
        for row_items in rows:
            # Сортируем элементы строки по X
            row_items = sorted(row_items, key=lambda x: x['center_x'])
            
            # Создаем ячейки для каждой колонки
            cells = [''] * num_columns
            
            for item in row_items:
                x_center = item['center_x']
                # Находим ближайшую колонку
                if column_positions:
                    closest_col_idx = min(range(num_columns), 
                                         key=lambda i: abs(x_center - column_positions[i]))
                    if abs(x_center - column_positions[closest_col_idx]) <= x_tolerance * 1.5:
                        # Объединяем текст, если в ячейке уже что-то есть
                        if cells[closest_col_idx]:
                            cells[closest_col_idx] += ' ' + str(item['text'])
                        else:
                            cells[closest_col_idx] = str(item['text'])
            
            # Убираем пустые строки (только пробелы)
            if any(cell.strip() for cell in cells):
                table_rows.append([cell.strip() for cell in cells])
        
        # Создаем DataFrame
        if table_rows and num_columns > 0:
            # Определяем заголовки
            headers = [f'Колонка_{i+1}' for i in range(num_columns)]
            
            # Проверяем первую строку - может быть заголовком
            if len(table_rows) > 1:
                first_row = table_rows[0]
                # Если первая строка содержит заголовко-подобный текст (короткие слова, без цифр)
                if first_row:
                    # Проверяем первые несколько ячеек
                    header_cells = [c for c in first_row[:3] if c]
                    is_header = all(
                        len(cell) < 50 and not cell.replace(',', '').replace('.', '').replace(' ', '').isdigit()
                        for cell in header_cells
                    ) if header_cells else False
                else:
                    is_header = False
                
                if is_header:
                    headers = first_row
                    table_df = pd.DataFrame(table_rows[1:], columns=headers)
                else:
                    table_df = pd.DataFrame(table_rows, columns=headers)
            else:
                table_df = pd.DataFrame(table_rows, columns=headers)
            
            structured[page_num] = table_df
        else:
            structured[page_num] = None
    
    return structured

def create_json_output(df, structured_tables, file_id):
    """Создает JSON файл со структурированными данными, оптимизированный для AI анализа"""
    json_filename = f'ocr_structured_{file_id}.json'
    json_path = os.path.join(app.config['UPLOAD_FOLDER'], json_filename)
    
    result = {
        'metadata': {
            'total_pages': len(df['page'].unique()),
            'total_text_blocks': len(df),
            'average_confidence': float(df['confidence'].mean()),
            'description': 'OCR результат финансового отчета. Используйте structured_table для анализа данных.'
        },
        'document_structure': {},
        'pages': []
    }
    
    for page_num in sorted(df['page'].unique()):
        page_data = df[df['page'] == page_num]
        
        # Определяем тип документа по тексту
        all_text = ' '.join(page_data['text'].astype(str)).lower()
        doc_type = 'unknown'
        if 'оборотно-сальдовая' in all_text or 'оборотная' in all_text:
            doc_type = 'trial_balance'
        elif 'баланс' in all_text:
            doc_type = 'balance_sheet'
        elif 'отчет' in all_text:
            doc_type = 'report'
        
        page_info = {
            'page_number': int(page_num),
            'document_type': doc_type,
            'text_blocks': page_data[['x0', 'y0', 'x1', 'y1', 'text', 'confidence']].to_dict('records'),
        }
        
        # Добавляем структурированную таблицу (ГЛАВНОЕ для AI)
        if page_num in structured_tables and structured_tables[page_num] is not None:
            table_df = structured_tables[page_num]
            
            # Преобразуем в более удобный формат для AI
            structured_data = {
                'columns': table_df.columns.tolist(),
                'rows': [],
                'row_count': len(table_df),
                'column_count': len(table_df.columns),
                'data_format': 'table',
                'description': 'Структурированная таблица данных. Каждая строка в rows - это массив значений ячеек в порядке колонок.'
            }
            
            # Преобразуем строки в словари для лучшей читаемости
            for idx, row in table_df.iterrows():
                row_dict = {}
                for col in table_df.columns:
                    try:
                        scalar_val = table_df.at[idx, col]
                        if pd.isna(scalar_val):
                            row_dict[col] = ''
                        else:
                            row_dict[col] = str(scalar_val)
                    except:
                        val_str = str(row[col]) if row[col] is not None and str(row[col]) != 'nan' else ''
                        row_dict[col] = val_str
                
                values = []
                for col in table_df.columns:
                    try:
                        scalar_val = table_df.at[idx, col]
                        if pd.isna(scalar_val):
                            values.append('')
                        else:
                            values.append(str(scalar_val))
                    except:
                        val_str = str(row[col]) if row[col] is not None and str(row[col]) != 'nan' else ''
                        values.append(val_str)
                    
                    structured_data['rows'].append({
                        'row_index': int(idx),
                        'cells': row_dict,
                        'values': values
                    })
            
            page_info['structured_table'] = structured_data
            
            # Добавляем также простой массив массивов для простого парсинга
            page_info['structured_table_array'] = {
                'headers': table_df.columns.tolist(),
                'data': table_df.values.tolist()
            }
            
            # Определяем заголовки документа
            header_texts = []
            for _, row in page_data.iterrows():
                y = row['y0']
                # Заголовки обычно в верхней части (первые 400 пикселей)
                if y < 400:
                    header_texts.append({
                        'text': row['text'],
                        'position': {'x0': float(row['x0']), 'y0': float(row['y0'])},
                        'confidence': float(row['confidence'])
                    })
            
            page_info['document_headers'] = header_texts
        
        result['pages'].append(page_info)
    
    # Добавляем общую информацию о структуре документа
    if structured_tables:
        all_tables_info = []
        for page_num, table_df in structured_tables.items():
            if table_df is not None and not table_df.empty:
                all_tables_info.append({
                    'page': int(page_num),
                    'columns': table_df.columns.tolist(),
                    'row_count': len(table_df)
                })
        result['document_structure'] = {
            'has_structured_tables': True,
            'tables': all_tables_info
        }
    else:
        result['document_structure'] = {
            'has_structured_tables': False
        }
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    return json_path

def create_csv_output(structured_tables, file_id):
    """Создает CSV файл со структурированными данными (объединение всех страниц)"""
    csv_filename = f'ocr_structured_{file_id}.csv'
    csv_path = os.path.join(app.config['UPLOAD_FOLDER'], csv_filename)
    
    all_tables = []
    for page_num in sorted(structured_tables.keys()):
        table_df = structured_tables[page_num]
        if table_df is not None and not table_df.empty:
            # Добавляем колонку с номером страницы
            table_df_with_page = table_df.copy()
            table_df_with_page.insert(0, 'Страница', page_num)
            all_tables.append(table_df_with_page)
    
    if all_tables:
        combined_df = pd.concat(all_tables, ignore_index=True)
        combined_df.to_csv(csv_path, index=False, encoding='utf-8-sig')  # utf-8-sig для Excel
    else:
        # Создаем пустой CSV файл
        pd.DataFrame().to_csv(csv_path, index=False, encoding='utf-8-sig')
    
    return csv_path

def create_html_visualization(df, page_dimensions, file_id):
    """Создает HTML файл с визуализацией текста по координатам"""
    html_filename = f'ocr_result_{file_id}.html'
    html_path = os.path.join(app.config['UPLOAD_FOLDER'], html_filename)
    
    # Группируем данные по страницам
    pages_html = []
    for page_num in sorted(df['page'].unique()):
        page_data = df[df['page'] == page_num]
        dims = page_dimensions.get(page_num, {'width': 2100, 'height': 2970})  # A4 по умолчанию
        
        # Масштабируем для отображения (уменьшаем для удобства просмотра)
        scale = 0.5
        page_width = dims['width'] * scale
        page_height = dims['height'] * scale
        
        text_elements = []
        for _, row in page_data.iterrows():
            x0 = row['x0'] * scale
            y0 = row['y0'] * scale
            x1 = row['x1'] * scale
            y1 = row['y1'] * scale
            text = str(row['text']).replace('<', '&lt;').replace('>', '&gt;').replace('&', '&amp;')
            conf = row['confidence']
            
            # Цвет фона в зависимости от уверенности
            if conf >= 0.9:
                bg_color = 'rgba(200, 255, 200, 0.3)'
            elif conf >= 0.7:
                bg_color = 'rgba(255, 255, 200, 0.3)'
            else:
                bg_color = 'rgba(255, 200, 200, 0.3)'
            
            width = x1 - x0
            height = y1 - y0
            
            text_elements.append(f'''
                <div class="text-block" style="
                    position: absolute;
                    left: {x0}px;
                    top: {y0}px;
                    width: {width}px;
                    height: {height}px;
                    background: {bg_color};
                    border: 1px solid rgba(0,0,0,0.2);
                    padding: 2px;
                    font-size: {max(8, height - 4)}px;
                    overflow: hidden;
                " title="Уверенность: {conf:.2%}">
                    {text}
                </div>
            ''')
        
        pages_html.append(f'''
            <div class="page-container">
                <h2>Страница {page_num}</h2>
                <div class="page" style="width: {page_width}px; height: {page_height}px; position: relative; border: 2px solid #333; margin: 20px auto; background: white;">
                    {''.join(text_elements)}
                </div>
            </div>
        ''')
    
    html_content = f'''<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>OCR Результат - Визуализация</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #f5f5f5;
            padding: 20px;
        }}
        .header {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .header h1 {{
            color: #333;
            margin-bottom: 10px;
        }}
        .header p {{
            color: #666;
        }}
        .controls {{
            background: white;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .controls button {{
            background: #667eea;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            margin-right: 10px;
            font-size: 14px;
        }}
        .controls button:hover {{
            background: #5568d3;
        }}
        .page-container {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .page-container h2 {{
            color: #333;
            margin-bottom: 15px;
            text-align: center;
        }}
        .text-block {{
            white-space: nowrap;
            line-height: 1.2;
        }}
        .legend {{
            background: white;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .legend h3 {{
            margin-bottom: 10px;
            color: #333;
        }}
        .legend-item {{
            display: inline-block;
            margin-right: 20px;
            margin-bottom: 5px;
        }}
        .legend-color {{
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 1px solid #ccc;
            margin-right: 5px;
            vertical-align: middle;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>📄 Результат OCR распознавания</h1>
        <p>Текст расположен согласно координатам из PDF файла</p>
    </div>
    
    <div class="legend">
        <h3>Легенда:</h3>
        <div class="legend-item">
            <span class="legend-color" style="background: rgba(200, 255, 200, 0.3);"></span>
            Высокая уверенность (≥90%)
        </div>
        <div class="legend-item">
            <span class="legend-color" style="background: rgba(255, 255, 200, 0.3);"></span>
            Средняя уверенность (70-90%)
        </div>
        <div class="legend-item">
            <span class="legend-color" style="background: rgba(255, 200, 200, 0.3);"></span>
            Низкая уверенность (&lt;70%)
        </div>
    </div>
    
    {''.join(pages_html)}
    
    <script>
        // Добавляем возможность масштабирования
        document.addEventListener('wheel', function(e) {{
            if (e.ctrlKey || e.metaKey) {{
                e.preventDefault();
                const pages = document.querySelectorAll('.page');
                pages.forEach(page => {{
                    const currentScale = parseFloat(page.style.transform.replace('scale(', '').replace(')', '')) || 1;
                    const newScale = e.deltaY > 0 ? currentScale * 0.9 : currentScale * 1.1;
                    page.style.transform = `scale(${{newScale}})`;
                    page.style.transformOrigin = 'top center';
                }});
            }}
        }});
    </script>
</body>
</html>
    '''
    
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    return html_path

@app.route('/download/excel/<filename>')
def download_excel(filename):
    """Скачивание Excel файла"""
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if os.path.exists(file_path):
        return send_file(
            file_path,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True,
            download_name='parsed_table_easyocr.xlsx'
        )
    return jsonify({'error': 'Файл не найден'}), 404

@app.route('/view/html/<filename>')
def view_html(filename):
    """Просмотр HTML файла"""
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    return jsonify({'error': 'Файл не найден'}), 404

@app.route('/download/json/<filename>')
def download_json(filename):
    """Скачивание JSON файла"""
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if os.path.exists(file_path):
        return send_file(
            file_path,
            mimetype='application/json',
            as_attachment=True,
            download_name='ocr_structured.json'
        )
    return jsonify({'error': 'Файл не найден'}), 404

@app.route('/download/csv/<filename>')
def download_csv(filename):
    """Скачивание CSV файла"""
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if os.path.exists(file_path):
        return send_file(
            file_path,
            mimetype='text/csv',
            as_attachment=True,
            download_name='ocr_structured.csv'
        )
    return jsonify({'error': 'Файл не найден'}), 404

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(debug=True, host='0.0.0.0', port=port)

