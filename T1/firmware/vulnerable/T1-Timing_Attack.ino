// T1 - Firmware vulnerable - Timing Attack Demo

const char SECRET[] = "S3cr3tK3y_2024!!"; 
const int  SECRET_LEN = 16;

bool check_vulnerable(const char* guess) {
  for (int i = 0; i < SECRET_LEN; i++) {
    if (guess[i] != SECRET[i]) return false;
    delayMicroseconds(200);  // amplifie la fuite
  }
  return true;
}

void setup() {
  Serial.begin(115200);
  while (!Serial);
}

void loop() {
  if (Serial.available() >= SECRET_LEN) {
    char buf[16];
    Serial.readBytes(buf, SECRET_LEN);
    bool ok = check_vulnerable(buf);
    Serial.write(ok ? 0x01 : 0x00);
  }
}
