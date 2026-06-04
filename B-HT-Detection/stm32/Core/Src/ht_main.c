/**
 * ht_main.c — Hardware Trojan Detector : firmware STM32 Nucleo-F401RE
 *
 * À intégrer dans un projet STM32CubeIDE généré avec :
 *   - USART2 : 115200 bauds, 8N1, DMA RX/TX
 *   - GPIO PA5 : LED LD2 (output)
 *   - GPIO PC13 : bouton USER (input)
 *   - TIM2 : 1ms timebase
 *   - FPU activé (Cortex-M4 avec FPU)
 *
 * Copiez ce fichier dans Core/Src/ et appelez ht_main() depuis main()
 * après MX_USART2_UART_Init() et MX_GPIO_Init().
 */
#include "ht_protocol.h"
#include "ht_detector_stm32.h"   /* poids + scaler + ht_predict() — copier depuis results/ vers Core/Inc/ */
#include "main.h"
#include <string.h>
#include <stdio.h>

/* ── Handles HAL (définis dans main.c par CubeMX) ────────────────── */
extern UART_HandleTypeDef huart2;
extern TIM_HandleTypeDef  htim2;

/* ── Buffers UART ────────────────────────────────────────────────── */
static uint8_t rx_header[HT_REQ_HEADER_SIZE];
static uint8_t rx_features_raw[HT_FEATURES_BYTES];
static float   features[HT_N_FEATURES];
static uint8_t tx_buf[HT_RESP_SIZE];

/* ── LED LD2 (PA5) ───────────────────────────────────────────────── */
#define LED_PORT   GPIOA
#define LED_PIN    GPIO_PIN_5

/* ── Compteur millisecondes ──────────────────────────────────────── */
static volatile uint32_t ms_tick = 0;

/* ── Prototypes internes ──────────────────────────────────────────── */
static void     ht_led_set(HT_Class class_id);
static void     ht_send_response(HT_Class class_id, float confidence,
                                  HT_Risk risk, uint16_t latency_ms);
static void     ht_send_info(void);
static void     ht_send_pong(void);
static uint32_t ht_get_ms(void);
static void     ht_error_blink(void);


/* ════════════════════════════════════════════════════════════════════
 * ht_main() — boucle principale à appeler depuis main()
 * ════════════════════════════════════════════════════════════════════ */
void ht_main(void) {

    /* Signal de démarrage : 3 clignotements rapides */
    for (int i = 0; i < 3; i++) {
        HAL_GPIO_WritePin(LED_PORT, LED_PIN, GPIO_PIN_SET);
        HAL_Delay(100);
        HAL_GPIO_WritePin(LED_PORT, LED_PIN, GPIO_PIN_RESET);
        HAL_Delay(100);
    }

    /* Envoyer message de bienvenue */
    const char* welcome = "\r\n[HT-DETECTOR] STM32 Nucleo-F401RE ready.\r\n"
                          "Architecture: 500->64->32->3 (TinyMLP int8)\r\n"
                          "Protocol: 115200 8N1 | Awaiting features...\r\n";
    HAL_UART_Transmit(&huart2, (uint8_t*)welcome, strlen(welcome), 100);

    /* ── Boucle principale ─────────────────────────────────────────── */
    while (1) {

        /* 1. Attendre l'en-tête de requête */
        HAL_StatusTypeDef status = HAL_UART_Receive(
            &huart2, rx_header, HT_REQ_HEADER_SIZE, HAL_MAX_DELAY
        );

        if (status != HAL_OK) {
            ht_error_blink();
            continue;
        }

        /* 2. Vérifier magic byte */
        if (rx_header[0] != HT_PROTO_REQ_MAGIC) {
            /* Flush : lire jusqu'au prochain octet valide */
            uint8_t dummy;
            while (HAL_UART_Receive(&huart2, &dummy, 1, 10) == HAL_OK);
            continue;
        }

        uint8_t  cmd    = rx_header[1];
        uint16_t n_feat = ((uint16_t)rx_header[2] << 8) | rx_header[3];

        /* 3. Dispatcher selon la commande */
        switch (cmd) {

        /* ── CMD_PING ───────────────────────────────────────────────── */
        case HT_CMD_PING:
            ht_send_pong();
            break;

        /* ── CMD_INFO ───────────────────────────────────────────────── */
        case HT_CMD_INFO:
            ht_send_info();
            break;

        /* ── CMD_PREDICT ────────────────────────────────────────────── */
        case HT_CMD_PREDICT:
            if (n_feat != HT_N_FEATURES) {
                /* Nombre de features incorrect */
                uint8_t err[] = {0xEE, 0x01};
                HAL_UART_Transmit(&huart2, err, sizeof(err), 50);
                break;
            }

            /* 4. Recevoir les features (500 × float32 = 2000 bytes) */
            status = HAL_UART_Receive(
                &huart2, rx_features_raw, HT_FEATURES_BYTES, 5000
            );
            if (status != HAL_OK) {
                ht_error_blink();
                break;
            }

            /* 5. Recevoir et vérifier le checksum */
            uint8_t chk_rx;
            HAL_UART_Receive(&huart2, &chk_rx, 1, 100);
            uint8_t chk_calc = ht_checksum(rx_features_raw, HT_FEATURES_BYTES);
            if (chk_rx != chk_calc) {
                uint8_t err[] = {0xEE, 0x02};   /* checksum error */
                HAL_UART_Transmit(&huart2, err, sizeof(err), 50);
                break;
            }

            /* 6. Copier bytes → floats (little-endian, même architecture) */
            memcpy(features, rx_features_raw, HT_FEATURES_BYTES);

            /* 7. Inférence Tiny MLP */
            uint32_t t_start = ht_get_ms();
            float    confidence;
            int      class_id = ht_predict(features, &confidence);
            uint32_t t_end    = ht_get_ms();
            uint16_t latency  = (uint16_t)(t_end - t_start);

            HT_Risk risk = (class_id == HT_TROJAN_TRIGGERED) ? HT_RISK_ALERT :
                           (class_id == HT_TROJAN_ENABLED)   ? HT_RISK_WARNING :
                                                               HT_RISK_OK;

            /* 8. Contrôler la LED */
            ht_led_set((HT_Class)class_id);

            /* 9. Envoyer la réponse */
            ht_send_response((HT_Class)class_id, confidence, risk, latency);
            break;

        default:
            /* Commande inconnue */
            {
                uint8_t err[] = {0xEE, 0xFF};
                HAL_UART_Transmit(&huart2, err, sizeof(err), 50);
            }
            break;
        }
    }
}


/* ════════════════════════════════════════════════════════════════════
 * Fonctions internes
 * ════════════════════════════════════════════════════════════════════ */

static void ht_led_set(HT_Class class_id) {
    switch (class_id) {
    case HT_TROJAN_DISABLED:
        HAL_GPIO_WritePin(LED_PORT, LED_PIN, GPIO_PIN_RESET);  /* OFF */
        break;
    case HT_TROJAN_ENABLED:
        /* Clignotement rapide 500ms */
        HAL_GPIO_TogglePin(LED_PORT, LED_PIN);
        break;
    case HT_TROJAN_TRIGGERED:
        HAL_GPIO_WritePin(LED_PORT, LED_PIN, GPIO_PIN_SET);    /* ON fixe */
        break;
    }
}

static void ht_send_response(HT_Class class_id, float confidence,
                               HT_Risk risk, uint16_t latency_ms) {
    uint8_t buf[HT_RESP_SIZE];
    uint8_t conf_bytes[4];
    memcpy(conf_bytes, &confidence, 4);   /* float32 little-endian */

    buf[0] = HT_PROTO_RESP_MAGIC;
    buf[1] = (uint8_t)class_id;
    buf[2] = conf_bytes[0];
    buf[3] = conf_bytes[1];
    buf[4] = conf_bytes[2];
    buf[5] = conf_bytes[3];
    buf[6] = (uint8_t)risk;
    buf[7] = (uint8_t)(latency_ms >> 8);
    buf[8] = (uint8_t)(latency_ms & 0xFF);
    buf[9] = ht_checksum(buf + 1, 8);    /* checksum sur bytes 1-8 */

    HAL_UART_Transmit(&huart2, buf, HT_RESP_SIZE, 100);
}

static void ht_send_info(void) {
    char info[256];
    int  len = snprintf(info, sizeof(info),
        "\r\n=== HT-DETECTOR INFO ===\r\n"
        "Model     : TinyMLP (KD from AT_T800)\r\n"
        "Arch      : %d->%d->%d->%d\r\n"
        "Accuracy  : %.2f%%\r\n"
        "Flash     : ~33KB / 512KB\r\n"
        "SRAM act. : ~2KB / 96KB\r\n"
        "MCU       : STM32F401RE @ 84MHz\r\n"
        "=========================\r\n",
        HT_N_INPUT, HT_HIDDEN1, HT_HIDDEN2, HT_N_CLASSES,
        HT_ACCURACY * 100.0f
    );
    HAL_UART_Transmit(&huart2, (uint8_t*)info, len, 200);
}

static void ht_send_pong(void) {
    uint8_t pong[] = {0xBB, HT_CMD_PING, 0x00};
    pong[2] = ht_checksum(pong + 1, 1);
    HAL_UART_Transmit(&huart2, pong, sizeof(pong), 50);
}

static uint32_t ht_get_ms(void) {
    return HAL_GetTick();
}

static void ht_error_blink(void) {
    for (int i = 0; i < 5; i++) {
        HAL_GPIO_TogglePin(LED_PORT, LED_PIN);
        HAL_Delay(50);
    }
}
