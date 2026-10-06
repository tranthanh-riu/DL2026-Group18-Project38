# Quy tắc viết code (người đọc là sinh viên năm 3, mới học Deep Learning)

Dự án: đề #38 Deep Time-Series Anomaly Detection (CWRU Bearing), nhóm 18.
Làm đúng theo docs/HUONG_DAN.md: đúng tên file, tên hàm, tên file .npz, tên cột CSV.
Không thêm tính năng ngoài hướng dẫn.

Dữ liệu đã được tải thủ công vào data/raw/ (97.mat ... 133.mat).
KHÔNG viết download.py và KHÔNG tải lại dữ liệu.

## Phong cách code
- Python đơn giản như sinh viên viết. Ưu tiên rõ ràng hơn gọn.
- Mỗi hàm làm MỘT việc, tối đa khoảng 30 dòng.
- Tên biến dài và rõ nghĩa (train_windows, val_scores), không viết tắt khó hiểu.
- KHÔNG dùng: decorator, lambda dài, comprehension lồng nhau, dataclass,
  type hint phức tạp, class (trừ nn.Module), kỹ thuật "thông minh" khó đọc.
- Vòng for thường thay cho trick một dòng, trừ khi trick đó là phép numpy rất ngắn.
- Mọi tham số (window, stride, seed, lr, batch, SNR...) đọc từ configs/default.yaml,
  không gõ cứng trong code.
- Mỗi file chạy được bằng: python -m src.<tên_module> (có hàm main()).

## Comment
- Viết comment tiếng Việt, đặt trước mỗi khối code, giải thích VÌ SAO chứ không chỉ LÀM GÌ.
- Ghi shape của mảng/tensor trong comment ở những chỗ quan trọng, ví dụ: # (N, 1024).

## Kiểm tra
- Sau mỗi bước quan trọng, print ra thứ để tôi kiểm tra bằng mắt:
  shape, số window theo tải/nhãn, giá trị loss, ngưỡng tìm được...
- Có phần "Tự kiểm tra" trong HUONG_DAN.md thì viết code in ra đúng các số đó.

## Cách làm việc
- Làm TUẦN TỰ từng bước (Bước 0, 1, 2...). Làm xong một bước thì DỪNG,
  để tôi chạy và xem output trước khi sang bước sau.
- Sau mỗi file viết xong, giải thích ngắn 3-5 dòng: file này làm gì, hàm nào quan trọng nhất.
  Tôi cần hiểu từng dòng để viết report và trả lời vấn đáp.
- Nếu hướng dẫn mơ hồ hoặc dữ liệu không như dự kiến (ví dụ độ dài tín hiệu,
  sampling rate lệch, tên biến không khớp số file), hãy hỏi tôi, đừng tự đoán.
- Không push file .mat, .npz, .pt. Commit nhỏ, mỗi bước một commit.
