# 1. Các bước triển khai
## Đọc và tiền xử lý dữ liệu CoNLL-U
- Hàm load_conllu được xây dựng để:
  - Loại bỏ các dòng chú thích và dòng rỗng.
  - Tách từng câu dựa trên dòng trống.
  - Trích xuất từ (word) và nhãn từ loại (UPOS) từ mỗi dòng hợp lệ.
  - Lưu dữ liệu dưới dạng danh sách các câu, trong đó mỗi câu là một danh sách các cặp (word, tag).

## Xây dựng từ điển từ vựng và nhãn
- Từ dữ liệu huấn luyện, hai từ điển được tạo:
  - word_to_ix: ánh xạ mỗi từ sang một chỉ số nguyên. Một token đặc biệt <UNK> được thêm vào để biểu diễn các từ chưa xuất hiện trong tập huấn luyện.
  - tag_to_ix: ánh xạ mỗi nhãn từ loại (UPOS) sang một chỉ số nguyên.

## Xây dựng lớp Dataset cho POS tagging
- Lớp POSDataset kế thừa từ torch.utils.data.Dataset, có nhiệm vụ:
  - Chuyển từng câu thành một chuỗi chỉ số từ (word indices) và chuỗi chỉ số nhãn (tag indices).
  - Trả về các tensor có độ dài khác nhau tương ứng với từng câu.

## Padding theo batch với collate_fn
- Do các câu có độ dài khác nhau, hàm collate_fn được sử dụng trong DataLoader để:
  - Padding các câu trong cùng một batch về cùng độ dài.
  - Sử dụng giá trị 0 cho padding, đồng thời đảm bảo các token padding sẽ không ảnh hưởng đến quá trình huấn luyện.
    
## Kiến trúc mô hình RNN cho gán nhãn chuỗi
- Mô hình SimpleRNNForTokenClassification được xây dựng với ba thành phần chính:
  - Embedding layer: ánh xạ mỗi từ sang một vector đặc trưng liên tục.
  - Recurrent Neural Network (RNN): học thông tin ngữ cảnh theo chuỗi từ trong câu.
  - Fully Connected layer: ánh xạ đầu ra của RNN sang không gian nhãn từ loại.

## Huấn luyện mô hình
- Quá trình huấn luyện sử dụng:
  - Hàm mất mát CrossEntropyLoss với tham số ignore_index=0 để bỏ qua các token padding.
  - Optimizer Adam để cập nhật trọng số mô hình.

## Đánh giá mô hình
- Hàm evaluate được xây dựng để:
  - So sánh nhãn dự đoán với nhãn thật tại từng token.
  - Chỉ tính toán độ chính xác trên các token không phải padding.
- Độ chính xác (accuracy) được sử dụng làm chỉ số đánh giá hiệu năng của mô hình.

## Dự đoán trên câu mới
- Hàm predict_sentence cho phép áp dụng mô hình đã huấn luyện lên một câu đầu vào bất kỳ:
  - Câu được tách từ và mã hóa thành chỉ số.
  - Mô hình dự đoán nhãn từ loại cho từng từ.
  - Kết quả trả về là danh sách các cặp (word, predicted_tag).

# 2. Kết quả chạy
- Chạy chương trình tại "Lab_05/src/lab5_pos_tagging.ipynb"
- Kết quả chạy:
  - Độ chính xác trên tập dev: 0.8763
  - Ví dụ dự đoán câu mới:
    - Câu: “I love NLP”
    - Dự đoán: [('I', 'PRON'), ('love', 'VERB'), ('NLP', 'NUM')]
   
# 3. Phân tích kết quả
## Độ chính xác trên tập Dev
- Mô hình RNN gán nhãn từ loại đạt độ chính xác 0.8763 trên tập dev, cho thấy mô hình đã học được các mẫu ngữ pháp cơ bản trong dữ liệu huấn luyện:
  - Mô hình có khả năng nhận diện tương đối tốt các nhãn từ loại phổ biến như PRON, VERB, NOUN, DET,…
  - Thông tin ngữ cảnh tuần tự do RNN học được giúp cải thiện dự đoán so với các phương pháp gán nhãn độc lập từng từ.
  - Việc bỏ qua các token padding trong quá trình tính loss và accuracy giúp kết quả đánh giá phản ánh đúng hiệu năng thực tế của mô hình.

## Dự đoán trên câu mới
- Nhận xét:
  - Từ “I” được dự đoán là PRON (đại từ) -> kết quả chính xác.
  - Từ “love” được dự đoán là VERB (động từ) -> kết quả chính xác.
  - Từ “NLP” bị dự đoán là NUM (số) -> đây là một dự đoán không chính xác, vì nhãn đúng nên là PROPN hoặc NOUN.
- Nguyên nhân:
  - Hạn chế của RNN đơn hướng: Mô hình chỉ khai thác ngữ cảnh từ trái sang phải, chưa tận dụng đầy đủ thông tin hai chiều của câu.
