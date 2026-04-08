การวิจัยและ Refactor ระบบ LLM Inference Engine (64-bit & GPU)
ผู้พัฒนา: Narawit Suraphan | ภาษาที่ใช้: C

ในการพัฒนาครั้งนี้ ผมได้ปรับปรุงโค้ดต้นฉบับของ llama2.c ครั้งใหญ่ โดยไม่ได้แค่แก้ให้รันได้ แต่เป็นการ ยกโครงสร้างระบบใหม่ทั้งหมด เพื่อให้รองรับมาตรฐานโมเดลปัจจุบัน และดึงพลังฮาร์ดแวร์ออกมาใช้งานให้ได้เต็มประสิทธิภาพที่สุด

1. 🪟 การแก้ปัญหา Windows API และ Cross-Platform
โค้ดเดิมออกแบบมาสำหรับ Linux (POSIX) เมื่อนำมาคอมไพล์บน Windows (MinGW) จึงพบปัญหาหลายจุด

แนวทางแก้ไข:

ปรับชนิดข้อมูลของ VirtualProtect จาก uint32_t* เป็น DWORD* ให้ตรงกับ Windows API

เพิ่ม Guard #ifndef __MINGW32__ เพื่อป้องกันการประกาศฟังก์ชัน clock_gettime ซ้ำซ้อน

ตั้งค่า Console ให้รองรับภาษาไทย (UTF-8) ด้วยคำสั่ง system("chcp 65001 > nul");

ผลลัพธ์: สามารถ Build และรันบนสภาพแวดล้อม Windows ได้อย่างเสถียร ไร้ Error

2. 🎯 การยกระดับเป็น 64-bit Double Precision
ส่วนนี้ถือเป็นแกนหลักของการปรับปรุงครั้งนี้เลยครับ

สิ่งที่ทำ:

ใช้ typedef double real_t; แทนที่ float ทั้งระบบ

เปลี่ยนฟังก์ชันคณิตศาสตร์ (Math Functions) ทั้งหมด เช่น sqrtf → sqrt, expf → exp, fabsf → fabs

ปรับ Memory Alignment ด้วย _aligned_malloc (64-byte)

ผลลัพธ์: ลดปัญหา Accumulated Rounding Error ใน Deep Layers ทำให้ Inference มีเสถียรภาพมากขึ้น อาการ Hallucination (AI ตอบวนลูป) ลดลงอย่างเห็นได้ชัด โดยเฉพาะในเคสที่ต้องวนลูปยาวๆ

3. 📦 พัฒนาระบบ GGUF v3 Parser
จากเดิมที่ระบบอ่านได้แค่ไฟล์ .bin ธรรมดา ตอนนี้พัฒนาให้รองรับไฟล์ GGUF ซึ่งเป็นมาตรฐานใหม่

การทำงาน (Implementation):

อ่าน Metadata แบบ Key-Value จาก Header ของไฟล์

ดึงค่าพารามิเตอร์สำคัญมาตั้งค่า เช่น llama.embedding_length, llama.block_count, llama.attention.head_count

ใช้ uint64_t สำหรับ File Offset เพื่อให้รองรับโมเดลที่มีขนาดใหญ่กว่า 4GB ได้

จุดสำคัญ: ต้องจัดการ Alignment 32-byte ให้ถูกต้องเป๊ะๆ เพราะถ้าคำนวณ Offset ผิด โปรแกรมจะเกิด Segfault ทันที

4. 🗜️ ระบบ Dequantization (Q4_0 → FP64)
เพื่อให้โมเดลที่ถูกบีบอัดมา สามารถนำมาประมวลผลร่วมกับระบบ FP64 ได้

แนวทาง: ดึงข้อมูลระดับ 4-bit (Nibble) ออกมาจาก Weights นำมาคูณกับค่า Scale (Delta) แล้วแปลง (Cast) เป็น double ก่อนส่งเข้า Pipeline การคำนวณ

ผลลัพธ์: ช่วยลดขนาดโมเดลลงได้ถึง ~8 เท่า โดยที่ยังรักษาความแม่นยำให้อยู่ในระดับที่สามารถใช้งานจริงได้สบายๆ

5. ⚡ GPU Acceleration ด้วย OpenCL (AMD RX 570)
โฟกัสที่การเร่งความเร็วการคูณเมทริกซ์ (MatMul) ซึ่งเป็นงานที่หนักที่สุดของ Inference

สิ่งที่ทำ:

โหลดไฟล์ opencl.dll แบบ Runtime (ใช้ LoadLibrary + GetProcAddress) ทำให้ไม่ต้องลง SDK ให้วุ่นวาย

เขียน OpenCL Kernel สำหรับทำ Matrix Multiplication โดยเฉพาะ

ใช้เทคนิค Loop Unrolling (Factor = 4) เพื่อรีดประสิทธิภาพ

เปิดใช้งาน FP64 บน GPU ด้วยโค้ด:

C
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
ผลลัพธ์: ระบบสามารถตรวจจับ GPU (ชิป Ellesmere / RX 570) ได้ทันที! ช่วยเพิ่ม Throughput ในการคำนวณแบบขนานได้อย่างชัดเจน

📝 สรุปผลการทำงาน
การปรับปรุงครั้งนี้เปลี่ยนระบบจาก Inference Engine แบบทดลอง ให้กลายเป็นระบบที่สามารถนำมาใช้งานได้จริง:

✅ ทันสมัย: รองรับโมเดลมาตรฐาน (GGUF)

✅ แม่นยำ: ประมวลผลด้วยความละเอียดสูง (FP64)

✅ รวดเร็ว: ดึงพลัง GPU มาใช้งานได้เต็มประสิทธิภาพ (OpenCL)

💡 Note: โปรเจกต์ทั้งหมดพัฒนาด้วยภาษา C โดยเน้นการควบคุม Low-level Performance ด้วยตัวเองทั้งหมด
🛠️ Tools: มีการใช้ Google CLI เข้ามาช่วยในการทำงาน
