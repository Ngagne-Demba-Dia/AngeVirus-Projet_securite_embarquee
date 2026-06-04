/**
 * ht_protocol.h — Protocole UART Hardware Trojan Detector
 * STM32 Nucleo-F401RE <-> Raspberry Pi 4 / PC
 *
 * Protocole binaire (115200 bauds, 8N1) :
 *
 * REQUEST (PC/RPi → STM32) :
 *   [0xAA] [CMD] [N_HIGH] [N_LOW] [features: N×float32 LE] [CHK]
 *
 * RESPONSE (STM32 → PC/RPi) :
 *   [0xBB] [CLASS] [CONF_3][CONF_2][CONF_1][CONF_0] [RISK] [LAT_H][LAT_L] [CHK]
 *
 * LED LD2 (PA5) :
 *   - OFF       : TrojanDisabled
 *   - 1Hz blink : TrojanEnabled
 *   - ON solid  : TrojanTriggered (ALERTE)
 */
#pragma once
#include <stdint.h>

/* ── Magic bytes ─────────────────────────────────────────────────── */
#define HT_PROTO_REQ_MAGIC   0xAA
#define HT_PROTO_RESP_MAGIC  0xBB

/* ── Commandes ───────────────────────────────────────────────────── */
#define HT_CMD_PREDICT       0x01   /* Inférence sur 500 features */
#define HT_CMD_INFO          0x02   /* Informations modèle */
#define HT_CMD_PING          0x03   /* Test connexion */

/* ── Tailles ─────────────────────────────────────────────────────── */
#define HT_N_FEATURES        500
#define HT_FEATURES_BYTES    (HT_N_FEATURES * sizeof(float))   /* 2000 bytes */
#define HT_REQ_HEADER_SIZE   4     /* magic + cmd + N_HIGH + N_LOW */
#define HT_RESP_SIZE         10    /* magic + class + conf×4 + risk + lat×2 + chk */

/* ── Classes et risques ──────────────────────────────────────────── */
typedef enum {
    HT_TROJAN_DISABLED  = 0,
    HT_TROJAN_ENABLED   = 1,
    HT_TROJAN_TRIGGERED = 2,
} HT_Class;

typedef enum {
    HT_RISK_OK      = 0,
    HT_RISK_WARNING = 1,
    HT_RISK_ALERT   = 2,
} HT_Risk;

/* ── Structure de réponse ────────────────────────────────────────── */
typedef struct {
    HT_Class  class_id;
    float     confidence;
    HT_Risk   risk;
    uint16_t  latency_ms;
    uint8_t   checksum;
} HT_Response;

/* ── Calcul checksum (XOR simple) ────────────────────────────────── */
static inline uint8_t ht_checksum(const uint8_t* data, uint16_t len) {
    uint8_t chk = 0;
    for (uint16_t i = 0; i < len; i++) chk ^= data[i];
    return chk;
}
