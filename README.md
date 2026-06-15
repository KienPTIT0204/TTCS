# Hệ thống điểm danh sinh viên bằng nhận diện khuôn mặt

## 1. Thông tin sinh viên thực hiện

* **Sinh viên thực hiện:** `Cao Trung Kiên
* **Mã sinh viên:** `B23DCCN455
* **Lớp:** `D23CQCN07-B
* **Đề tài:** Xây dựng hệ thống điểm danh sinh viên bằng nhận diện khuôn mặt

---

## 2. Giới thiệu dự án

Đây là hệ thống web hỗ trợ điểm danh sinh viên bằng công nghệ nhận diện khuôn mặt. Hệ thống cho phép quản lý sinh viên, đăng ký dữ liệu khuôn mặt, tạo buổi học, mở camera điểm danh, tự động nhận diện sinh viên và lưu kết quả điểm danh vào cơ sở dữ liệu.

Project được xây dựng với mục tiêu giảm thời gian điểm danh thủ công, hạn chế sai sót trong quá trình ghi nhận chuyên cần và hỗ trợ giảng viên quản lý kết quả điểm danh một cách thuận tiện hơn.

Hệ thống được xây dựng dưới dạng ứng dụng web, có giao diện đơn giản, dễ sử dụng và phù hợp với môi trường demo phục vụ học tập, báo cáo cuối kỳ hoặc đồ án môn học.

---

## 3. Chức năng chính

* Đăng nhập và phân quyền người dùng.
* Quản lý danh sách sinh viên.
* Đăng ký khuôn mặt cho sinh viên bằng webcam.
* Huấn luyện và lưu trữ dữ liệu nhận diện khuôn mặt.
* Tạo và quản lý buổi học.
* Điểm danh sinh viên bằng camera.
* Kiểm tra liveness cơ bản bằng chuyển động đầu trái/phải.
* Tự động ghi nhận sinh viên vắng mặt khi đóng buổi học.
* Xem báo cáo điểm danh.
* Xuất báo cáo điểm danh ra file Excel.
* Sinh viên có thể xem lịch sử điểm danh cá nhân.

---

## 4. Công nghệ sử dụng

| Thành phần          | Công nghệ                     |
| ------------------- | ----------------------------- |
| Backend             | Python, Flask                 |
| Frontend            | HTML, CSS, JavaScript, Jinja2 |
| Cơ sở dữ liệu       | SQLite                        |
| Xử lý ảnh           | OpenCV                        |
| Nhận diện khuôn mặt | face_recognition              |
| Báo cáo             | Pandas, OpenPyXL              |

---

## 5. Cấu trúc thư mục

```text
face_attendance_demo/
├── app.py
├── requirements.txt
├── README.md
├── static/
│   └── css/
│       └── style.css
├── templates/
│   ├── base.html
│   ├── login.html
│   ├── dashboard.html
│   ├── students.html
│   ├── student_form.html
│   ├── enroll.html
│   ├── sessions.html
│   ├── session_form.html
│   ├── attendance_camera.html
│   ├── reports.html
│   ├── my_attendance.html
│   └── users.html
└── instance/
    ├── attendance.db
    ├── faces/
    ├── encodings.pkl
    ├── labels.json
    └── encoded_index.json
```

---

## 6. Hướng dẫn cài đặt và chạy chương trình

### Bước 1: Clone project

```bash
git clone <link-repository>
cd face_attendance_demo
```

### Bước 2: Tạo môi trường ảo

```bash
python -m venv .venv
```

### Bước 3: Kích hoạt môi trường ảo

Trên Windows:

```bash
.venv\Scripts\activate
```

Trên macOS hoặc Linux:

```bash
source .venv/bin/activate
```

### Bước 4: Cài đặt thư viện

```bash
pip install -r requirements.txt
```

### Bước 5: Chạy chương trình

```bash
python app.py
```

Sau đó mở trình duyệt và truy cập địa chỉ:

```text
http://127.0.0.1:5000
```

---

## 7. Hướng dẫn sử dụng cơ bản

### 7.1. Đăng nhập hệ thống

Người dùng truy cập trang đăng nhập và nhập tài khoản được cấp. Sau khi đăng nhập, hệ thống sẽ hiển thị giao diện phù hợp với vai trò của người dùng.

Các vai trò trong hệ thống gồm:

* `admin`: quản trị viên hệ thống.
* `teacher`: giảng viên.
* `student`: sinh viên.

### 7.2. Quản lý sinh viên

Admin hoặc Teacher có thể thực hiện các chức năng quản lý sinh viên như:

* Xem danh sách sinh viên.
* Thêm sinh viên mới.
* Xóa sinh viên.
* Xóa dữ liệu khuôn mặt của sinh viên.
* Truy cập chức năng đăng ký khuôn mặt cho sinh viên.

### 7.3. Đăng ký khuôn mặt

Để đăng ký khuôn mặt cho sinh viên:

1. Vào danh sách sinh viên.
2. Chọn sinh viên cần đăng ký khuôn mặt.
3. Mở giao diện đăng ký khuôn mặt.
4. Bật camera.
5. Chụp nhiều ảnh khuôn mặt của sinh viên.
6. Hệ thống kiểm tra ảnh và lưu dữ liệu khuôn mặt.
7. Sau khi có đủ dữ liệu, hệ thống cập nhật dữ liệu nhận diện.

Lưu ý: Khi chụp ảnh đăng ký, trong khung hình chỉ nên có một khuôn mặt để tránh lưu nhầm dữ liệu.

### 7.4. Tạo buổi học

Teacher có thể tạo buổi học bằng cách nhập các thông tin:

* Tên môn học.
* Lớp học.
* Ngày học.
* Thời gian bắt đầu.
* Thời gian kết thúc.

Sau khi tạo thành công, buổi học sẽ được lưu vào hệ thống và có thể sử dụng để mở phiên điểm danh bằng camera.

### 7.5. Điểm danh bằng camera

Để điểm danh sinh viên:

1. Teacher chọn buổi học cần điểm danh.
2. Mở giao diện camera điểm danh.
3. Sinh viên đưa khuôn mặt vào camera.
4. Sinh viên thực hiện xác minh liveness bằng cách quay nhẹ đầu trái/phải.
5. Hệ thống nhận diện khuôn mặt.
6. Nếu nhận diện thành công, hệ thống lưu kết quả điểm danh.

Nếu sinh viên đã điểm danh trước đó trong cùng một buổi học, hệ thống sẽ không lưu trùng mà hiển thị thông báo đã điểm danh.

### 7.6. Đóng buổi học

Sau khi kết thúc điểm danh, Teacher có thể đóng buổi học. Khi đóng buổi học, hệ thống sẽ tự động kiểm tra danh sách sinh viên của lớp. Những sinh viên chưa điểm danh sẽ được ghi nhận là vắng mặt.

### 7.7. Xem và xuất báo cáo

Người dùng có quyền có thể xem báo cáo điểm danh theo các tiêu chí như:

* Ngày học.
* Lớp học.
* Buổi học.

Hệ thống hỗ trợ xuất báo cáo điểm danh ra file Excel để phục vụ lưu trữ và thống kê.

---

## 8. Một số API chính

| API                               | Chức năng                       |
| --------------------------------- | ------------------------------- |
| `/login`                          | Đăng nhập hệ thống              |
| `/logout`                         | Đăng xuất                       |
| `/students`                       | Hiển thị danh sách sinh viên    |
| `/students/add`                   | Thêm sinh viên                  |
| `/students/<student_id>/enroll`   | Giao diện đăng ký khuôn mặt     |
| `/api/enroll/<student_id>`        | API xử lý đăng ký khuôn mặt     |
| `/api/train`                      | Huấn luyện dữ liệu nhận diện    |
| `/sessions`                       | Danh sách buổi học              |
| `/sessions/add`                   | Tạo buổi học                    |
| `/sessions/<session_id>/camera`   | Giao diện điểm danh bằng camera |
| `/api/recognize/<session_id>`     | API nhận diện và điểm danh      |
| `/api/close-session/<session_id>` | Đóng buổi học                   |
| `/reports`                        | Xem báo cáo điểm danh           |
| `/reports/export`                 | Xuất báo cáo Excel              |

---

## 9. Lưu ý khi đưa project lên GitHub

Không nên đưa các thư mục và file sau lên GitHub:

```text
.venv/
__pycache__/
instance/faces/
instance/encodings.pkl
instance/encoded_index.json
instance/labels.json
*.db
```

Các file trên có thể chứa dữ liệu khuôn mặt, dữ liệu sinh viên hoặc dữ liệu sinh trắc học cá nhân. Khi public project, nên loại bỏ các dữ liệu này để đảm bảo an toàn thông tin.

Có thể tạo file `.gitignore` với nội dung:

```gitignore
.venv/
__pycache__/
*.pyc
instance/faces/
instance/*.pkl
instance/*.json
instance/*.db
*.xlsx
```

---

## 10. Hạn chế của hệ thống

* Hệ thống mới chạy ở môi trường demo.
* Cơ sở dữ liệu SQLite phù hợp với quy mô nhỏ.
* Độ chính xác nhận diện phụ thuộc vào ánh sáng, camera và góc mặt.
* Chức năng liveness hiện tại còn ở mức cơ bản.
* Chưa triển khai trên server thực tế.
* Chưa tích hợp với hệ thống quản lý đào tạo của nhà trường.
* Chưa tối ưu hiệu năng khi số lượng sinh viên và dữ liệu khuôn mặt tăng cao.

---

## 11. Hướng phát triển

* Chuyển cơ sở dữ liệu từ SQLite sang PostgreSQL hoặc MySQL.
* Triển khai hệ thống lên server để sử dụng qua mạng.
* Bổ sung kiểm tra chớp mắt và phát hiện ảnh giả nâng cao.
* Tối ưu tốc độ nhận diện khi số lượng sinh viên lớn.
* Bổ sung chức năng quản lý lớp học chi tiết.
* Bổ sung chức năng sửa thông tin sinh viên và sửa buổi học.
* Tích hợp gửi thông báo kết quả điểm danh cho sinh viên.
* Xây dựng API để kết nối với hệ thống quản lý đào tạo.
* Cải thiện giao diện người dùng để hệ thống thân thiện và dễ sử dụng hơn.

