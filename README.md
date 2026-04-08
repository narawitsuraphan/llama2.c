รายละเอียดการวิจัยและการ Refactor ระบบ LLM Inference Engine (64-bit & GPU)
ผู้พัฒนา: Narawit Suraphan
ภาษา: C
ในการพัฒนาครั้งนี้ ผมได้ปรับปรุงโค้ดต้นฉบับของ llama2.c ครั้งใหญ่ โดยไม่ได้แค่แก้ให้รันได้ แต่เป็นการยกโครงสร้างระบบใหม่ทั้งหมด เพื่อให้รองรับมาตรฐานโมเดลปัจจุบัน และใช้ฮาร์ดแวร์ได้เต็มประสิทธิภาพ
1. การแก้ปัญหา Windows API และ Cross-Platform
โค้ดเดิมออกแบบมาสำหรับ Linux (POSIX) เมื่อคอมไพล์บน Windows (MinGW) พบปัญหาหลายจุด
แนวทางที่แก้:
ปรับชนิดข้อมูลของ VirtualProtect
จาก uint32_t* → DWORD* ให้ตรงกับ Windows API
เพิ่ม guard #ifndef __MINGW32__ ป้องกันการซ้ำของ clock_gettime
ตั้งค่า console เป็น UTF-8 ด้วย
system("chcp 65001 > nul");
ผลลัพธ์:
สามารถ build และรันบน Windows ได้เสถียรโดยไม่มี error
2. การยกระดับเป็น 64-bit Double Precision
ส่วนนี้เป็นแกนหลักของงาน
สิ่งที่ทำ:
ใช้ typedef double real_t; แทน float ทั้งระบบ
เปลี่ยน math functions:
sqrtf → sqrt
expf → exp
fabsf → fabs
ปรับ memory alignment ด้วย _aligned_malloc (64-byte)
ผลที่ได้:
ลด accumulated rounding error ใน deep layers
inference มีเสถียรภาพมากขึ้น
อาการ hallucination ลดลง โดยเฉพาะเคส loop
3. ระบบ GGUF v3 Parser
จากเดิมที่อ่านได้แค่ .bin ได้พัฒนาให้รองรับ GGUF ซึ่งเป็นมาตรฐานใหม่
Implementation:
อ่าน metadata แบบ Key-Value จาก header
ดึงค่า เช่น:
llama.embedding_length
llama.block_count
llama.attention.head_count
ใช้ uint64_t สำหรับ file offset รองรับโมเดล > 4GB
จุดสำคัญ:
ต้องจัดการ alignment 32-byte ให้ถูกต้อง
คำนวณ offset ผิดจะ segfault ทันที
4. ระบบ Dequantization (Q4_0 → FP64)
เพื่อให้โมเดลที่ถูกบีบอัดยังคงใช้งานกับ FP64 ได้
แนวทาง:
ดึงข้อมูล 4-bit (nibble) จาก weights
คูณ scale (delta)
แปลงเป็น double ก่อนเข้า pipeline
ผลลัพธ์:
ลดขนาดโมเดล ~8 เท่า
ความแม่นยำยังอยู่ในระดับที่ใช้งานจริงได้
5. GPU Acceleration ด้วย OpenCL (AMD RX 570)
โฟกัสที่การเร่ง MatMul ซึ่งเป็นงานหลักของ inference
สิ่งที่ทำ:
โหลด opencl.dll แบบ runtime (ไม่ต้องลง SDK)
ใช้ LoadLibrary + GetProcAddress
เขียน OpenCL kernel สำหรับ matrix multiplication
ใช้ loop unrolling (factor = 4)
เปิดใช้ FP64:
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
ผลลัพธ์:
ตรวจจับ GPU (Ellesmere / RX 570) ได้ทันที
เพิ่ม throughput ในงานคำนวณแบบขนานอย่างชัดเจน
สรุป
การปรับปรุงครั้งนี้เปลี่ยนระบบจาก inference engine แบบทดลอง ให้กลายเป็นระบบที่ใช้งานจริงได้:
รองรับโมเดลมาตรฐาน (GGUF)
เพิ่มความแม่นยำ (FP64)
ใช้ GPU ได้เต็มประสิทธิภาพ (OpenCL)
ทั้งหมดพัฒนาด้วย ภาษา C และเน้นควบคุม low-level performance ด้วยตัวเอง
tools ที่ช่วยทำ Google CLI
