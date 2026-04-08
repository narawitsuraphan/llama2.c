 Llama2.c GGUF 64-bit GPU Research Edition 🚀
  Developed by: NARWIT SURAPHAN

  โปรเจกต์นี้เป็นการวิจัยและพัฒนาการเพิ่มประสิทธิภาพระบบ LLM Inference โดยการปรับปรุงโครงสร้างจาก llama2.c (ต้นฉบับของ Andrej Karpathy) ให้รองรับมาตรฐานไฟล์ GGUF ในระดับความแม่นยำสูงสุด (64-bi                      it Double Precision)
  และเร่งความเร็วด้วย GPU AMD RX 570

  การปรับปรุงและแก้ไขโค้ดเชิงลึก (Technical Implementation)

  1. การยกเครื่องสถาปัตยกรรมเป็น 64-bit (Full 64-bit Precision)
  ผมได้เปลี่ยนระบบการคำนวณและจัดการหน่วยความจำทั้งหมดจาก 32-bit Float เป็น 64-bit Double เพื่อความแม่นยำเชิงตัวเลขสูงสุด:
   * Typedef real_t: ใช้การนิยามชนิดข้อมูลแบบยืดหยุ่นผ่าน typedef double real_t; เพื่อให้ระบบคำนวณทั้งประมวลผลบน Double Precision
   * Math Function Migration: แก้ไขฟังก์ชันคณิตศาสตร์พื้นฐานจากตระกูล Single-precision เป็น Double-precision ทั้งหมด เช่น:
       * sqrtf() (32-bit) $\rightarrow$ sqrt() (64-bit)
       * expf() (32-bit) $\rightarrow$ exp() (64-bit)
       * fabsf() (32-bit) $\rightarrow$ fabs() (64-bit)
   * Memory Alignment: ปรับการจองหน่วยความจำ (Memory Allocation) เป็นแบบ 64-byte aligned เพื่อให้สอดคล้องกับโครงสร้างข้อมูล 64-bit และการอ่านข้อมูลของ CPU/GPU รุ่นใหม่ ทำให้รับส่งข้อมูลได้รวดเ                      เร็วขึ้น

  2. ระบบอ่านไฟล์โมเดล GGUF (v3) Parser
  ผมได้เขียนตัวอ่านไฟล์ GGUF ขึ้นมาใหม่จากศูนย์ (Scratch) เพื่อให้หลุดพ้นจากข้อจำกัดของไฟล์โมเดลแบบเก่า:
   * Metadata KV Extraction: พัฒนาระบบวนลูปอ่าน Key-Value Pairs ของ GGUF เพื่อดึงค่า llama.embedding_length (Dimension), llama.block_count (Layers) มาตั้งค่าโมเดลโดยอัตโนมัติ
   * Tensor Mapping: พัฒนาระบบค้นหาชื่อ Tensor (เช่น token_embd.weight) และคำนวณตำแหน่ง Offset ในไฟล์แบบ 64-bit ทำให้สามารถโหลดน้ำหนัก (Weights) เข้าสู่หน่วยความจำได้อย่างแม่นยำ

  3. ระบบถอดรหัสข้อมูลบีบอัด (Q4_0 Dequantization)
  เนื่องจากไฟล์ GGUF ส่วนใหญ่ถูกบีบอัดมา ผมจึงเขียนฟังก์ชันถอดรหัสเฉพาะตัว:
   * Q4_0 to 64-bit: พัฒนาอัลกอริทึมในการดึงข้อมูล 4-bit (Nibbles) ออกจาก Super-block ของ GGUF แล้วทำการคำนวณร่วมกับค่า Delta (Scale) เพื่อแปลงกลับมาเป็น 64-bit Double ในแรมโดยตรง
     ทำให้เราใช้โมเดลขนาดใหญ่ในขณะที่ยังรักษาความแม่นยำสูงสุดไว้ได้

  4. การเร่งความเร็วด้วย AMD GPU (RX 570) ผ่าน OpenCL
  ผมได้พัฒนา GPU Backend เพื่อดึงพลังจาก AMD RX 570 มาช่วยประมวลผล:
   * Dynamic OpenCL Loader: ใช้เทคนิคการโหลด opencl.dll แบบ Runtime (Dynamic Loading) ทำให้โปรแกรมสามารถรันได้ทันทีโดย ไม่ต้องติดตั้ง OpenCL SDK
   * Optimized OpenCL Kernel: เขียนภาษา C สำหรับ GPU (Kernel) โดยใช้เทคนิค Loop Unrolling และ Vectorized Loading เพื่อให้การคำนวณ Matrix Multiplication (MatMul) ทำงานได้เต็มประสิทธิภาพบนสถาปัต           ตยกรรม Polaris
     ของ AMD
   * 64-bit GPU Compute: เปิดใช้งาน Extension cl_khr_fp64 ใน Kernel เพื่อให้การคำนวณบนการ์ดจอมีความแม่นยำระดับ 64-bit เท่ากับ CPU

  ผลลัพธ์ที่ได้ (Benefits)

   * Higher Precision: การใช้ 64-bit ช่วยลดสะสมของ Error ในการคำนวณระหว่าง Layer ทำให้ AI รักษาบริบท (Context) ได้ยาวขึ้นและแม่นยำกว่าระบบ 32-bit ทั่วไป
   * Modern Compatibility: รองรับไฟล์โมเดลยอดนิยมอย่าง GGUF ที่หาโหลดได้ทั่วไป
   * Hardware Empowerment: สามารถรันโมเดลภาษาขนาดใหญ่บนการ์ดจอรุ่นเก่าอย่าง RX 570 ได้อย่างลื่นไหลและมีประสิทธิภาพ

  ---

  วิธีการติดตั้งและรัน (Installation)

   1. คอมพลาย:

   1    gcc -Ofast -fopenmp -D_WIN32 -o run_gguf64.exe -I. run_gguf64.c win.c

   2. รันโมเดล:
   1    .\run_gguf64.exe <ชื่อโมเดล>.gguf

  Developed with ❤️ and Research by NARWIT SURAPHAN (2026)
