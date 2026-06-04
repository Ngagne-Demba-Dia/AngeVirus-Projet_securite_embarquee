# STM32 Nucleo-F401RE — Hardware Trojan Detector

## Configuration STM32CubeIDE

### 1. Nouveau projet CubeIDE
- Board : NUCLEO-F401RE
- Firmware : STM32Cube FW_F4

### 2. Configuration IOC (STM32CubeMX)

**USART2** (déjà connecté au ST-LINK = port USB/COM) :
- Mode : Asynchronous
- Baud rate : 115200
- Word length : 8 bits
- Parity : None
- Stop bits : 1
- DMA : désactivé (polling suffisant pour 2KB)

**GPIO PA5** (LED LD2) :
- Mode : GPIO Output Push Pull
- Label : LED_GREEN

**FPU** :
- Floating Point Unit : activé (Cortex-M4 avec FPU matérielle)

**Project Manager** :
- Generate peripheral initialization as a pair of .c/.h files per peripheral : ✓

### 3. Copier les fichiers dans le projet

```
STM32Project/
├── Core/
│   ├── Inc/
│   │   ├── ht_protocol.h        ← copier depuis stm32/Core/Inc/
│   │   ├── ht_main.h            ← copier depuis stm32/Core/Inc/
│   │   └── ht_detector_stm32.h  ← copier depuis results/
│   └── Src/
│       └── ht_main.c            ← copier depuis stm32/Core/Src/
```

### 4. Modifier main.c

Ajouter dans main.c (généré par CubeMX) :

```c
/* USER CODE BEGIN Includes */
#include "ht_main.h"
/* USER CODE END Includes */

/* Dans main(), après les initialisations : */
/* USER CODE BEGIN 2 */
ht_main();   /* ne retourne jamais */
/* USER CODE END 2 */
```

### 5. Options de compilation

Dans Project Properties → C/C++ Build → Settings → MCU GCC Compiler :
- Optimization : -O2
- Floating point : Hardware (FPU)

Ajouter `-lm` dans Linker flags pour la fonction expf().

### 6. Build et Flash

```
Project → Build Project
Run → Run As → STM32 Application
```

La LED LD2 clignote 3× au démarrage. Le port COM est prêt.

---

## Protocole UART

### Requête (PC/RPi → STM32)

```
[0xAA] [CMD] [N_HIGH] [N_LOW] [features: 2000 bytes] [CHK]
```

| Byte | Valeur | Description |
|------|--------|-------------|
| 0    | 0xAA   | Magic byte |
| 1    | 0x01   | CMD_PREDICT |
| 2    | 0x01   | N_HIGH (500 >> 8) |
| 3    | 0xF4   | N_LOW (500 & 0xFF) |
| 4-2003 | float32 LE × 500 | Features normalisées |
| 2004 | XOR    | Checksum des features |

### Réponse (STM32 → PC/RPi)

```
[0xBB] [CLASS] [CONF×4] [RISK] [LAT×2] [CHK]
```

| Byte | Description |
|------|-------------|
| 0    | 0xBB magic |
| 1    | Classe (0=Disabled, 1=Enabled, 2=Triggered) |
| 2-5  | Confidence float32 LE |
| 6    | Risque (0=OK, 1=WARNING, 2=ALERT) |
| 7-8  | Latence uint16 big-endian (ms) |
| 9    | Checksum XOR bytes 1-8 |

### Commandes disponibles

| CMD  | Code | Description |
|------|------|-------------|
| PING | 0x03 | Test connexion |
| INFO | 0x02 | Informations modèle |
| PREDICT | 0x01 | Inférence sur 500 features |

---

## LED LD2 (PA5)

| État Trojan | LED |
|-------------|-----|
| Disabled    | OFF |
| Enabled     | Clignotement 1Hz |
| Triggered   | ON fixe (ALERTE) |

---

## Performance attendue

- Latence inférence : ~1-5 ms @ 84MHz avec FPU matérielle
- Throughput : ~200-1000 inférences/seconde
- Consommation : ~30mA active, ~5mA sleep
