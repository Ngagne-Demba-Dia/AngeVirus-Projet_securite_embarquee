// T1 - Firmware vulnerable - Timing Attack Demo - version constant-time

const char SECRET[] = "S3cr3tK3y_2024!!"; 
const int  SECRET_LEN = 16;

// Version constant-time - protection contre timing attack
bool check_constant_time(const char* guess) {
  uint8_t diff = 0;
  for (int i = 0; i < SECRET_LEN; i++) {
    diff |= (uint8_t)(guess[i] ^ SECRET[i]);
    delayMicroseconds(200);  
  }
  return (diff == 0);
}


void setup() {
  Serial.begin(115200);
  while (!Serial);
}

void loop() {
  if (Serial.available() >= SECRET_LEN) {
    char buf[16];
    Serial.readBytes(buf, SECRET_LEN);
    bool ok = check_constant_time(buf);
    Serial.write(ok ? 0x01 : 0x00);
  }
}
