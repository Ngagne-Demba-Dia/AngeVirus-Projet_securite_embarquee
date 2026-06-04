/**
 * ht_main.h — Hardware Trojan Detector STM32
 */
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Boucle principale du détecteur HT.
 * Appelez depuis main() après initialisation des périphériques HAL.
 *
 * Exemple dans main.c :
 *   MX_GPIO_Init();
 *   MX_USART2_UART_Init();
 *   MX_TIM2_Init();
 *   ht_main();   // <-- ne retourne jamais (boucle infinie)
 */
void ht_main(void);

#ifdef __cplusplus
}
#endif
