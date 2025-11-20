#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
아두이노 원시 시리얼 통신 테스트
"""


import serial
import time


def test_arduino_raw(port='/dev/ttyACM0', baudrate=9600):
   """아두이노 원시 시리얼 통신 테스트"""
   try:
       print(f"🔌 아두이노 연결 시도: {port}")
       ser = serial.Serial(port=port, baudrate=baudrate, timeout=1)
       time.sleep(3)  # 아두이노 초기화 대기
       print(f"✅ 아두이노 연결 성공: {port}")
      
       # 시리얼 포트 정보 출력
       print(f"📊 시리얼 포트 정보:")
       print(f"  - 포트: {ser.port}")
       print(f"  - 보드레이트: {ser.baudrate}")
       print(f"  - 타임아웃: {ser.timeout}")
       print(f"  - 쓰기 타임아웃: {ser.write_timeout}")
      
       # 초기 응답 확인
       print(f"\n📥 초기 응답 확인 (5초 대기):")
       time.sleep(5)
       initial_responses = []
       while ser.in_waiting > 0:
           response = ser.readline().decode('utf-8', errors='ignore').strip()
           if response:
               initial_responses.append(response)
               print(f"  {response}")
      
       if not initial_responses:
           print("  ❌ 초기 응답 없음")
      
       # 단일 문자 테스트
       print(f"\n🔤 단일 문자 테스트:")
       test_chars = ['f', 's', 'l', 'r']
      
       for char in test_chars:
           print(f"📤 문자 전송: '{char}'")
           ser.write(char.encode('utf-8'))
           ser.flush()
          
           time.sleep(1)
           if ser.in_waiting > 0:
               response = ser.readline().decode('utf-8', errors='ignore').strip()
               print(f"📥 응답: {response}")
           else:
               print("❌ 응답 없음")
      
       # 문자열 명령 테스트
       print(f"\n📝 문자열 명령 테스트:")
       test_commands = ['STATUS', 'MOVE_FORWARD', 'STOP']
      
       for cmd in test_commands:
           print(f"📤 명령 전송: '{cmd}'")
          
           # 명령 전송
           ser.write(cmd.encode('utf-8'))
           ser.write(b'\r\n')  # CRLF로 전송
           ser.flush()
          
           # 응답 대기
           time.sleep(2)
          
           # 모든 응답 읽기
           responses = []
           while ser.in_waiting > 0:
               response = ser.readline().decode('utf-8', errors='ignore').strip()
               if response:
                   responses.append(response)
          
           if responses:
               print(f"📥 응답:")
               for resp in responses:
                   print(f"  {resp}")
           else:
               print("❌ 응답 없음")
      
       # 바이트 단위 테스트
       print(f"\n🔢 바이트 단위 테스트:")
       test_bytes = [b'f', b's', b'l', b'r']
      
       for byte_cmd in test_bytes:
           print(f"📤 바이트 전송: {byte_cmd}")
           ser.write(byte_cmd)
           ser.flush()
          
           time.sleep(1)
           if ser.in_waiting > 0:
               response = ser.readline().decode('utf-8', errors='ignore').strip()
               print(f"📥 응답: {response}")
           else:
               print("❌ 응답 없음")
      
       ser.close()
       print("\n✅ 테스트 완료")
      
   except Exception as e:
       print(f"❌ 오류: {e}")
       import traceback
       traceback.print_exc()


if __name__ == "__main__":
   test_arduino_raw('/dev/ttyACM0')

