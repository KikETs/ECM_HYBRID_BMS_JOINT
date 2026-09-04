/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2026 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "main.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include <string.h>
#include "sop_core.h"
#include "sop_tables.h"
#include "soh_core.h"
#include "soh_simd.h"
#include "soh_tables.h"
/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */
#define CEMA_PROTOCOL_MAGIC 0x43454D41UL
#define CEMA_PROTOCOL_VERSION 5UL
#define CEMA_CMD_QUERY 0x51U
#define CEMA_CMD_RESET 0x52U
#define CEMA_CMD_SAMPLE 0x53U
#define CEMA_CMD_SAMPLE_DT 0x44U
/* SOP 벤치 명령.  단계를 따로 재야 무엇이 지배하는지 보인다. */
#define SOP_CMD_QUERY   0x60U   /* 보드/자산 정보 */
#define SOP_CMD_REFF    0x61U   /* 표 조회 1 회 */
#define SOP_CMD_TRIM    0x62U   /* 12->2 선형 1 회 */
#define SOP_CMD_SOLVE   0x63U   /* 고정점 반전 (표 조회 N 회) */
#define SOP_CMD_FULL    0x64U   /* 트림 + 반전 = SOP 1 회 */
#define SOP_CMD_FEAT    0x65U   /* 특징 갱신 1 샘플 */
#define SOP_CMD_EKF     0x66U   /* SOC EKF 1 스텝 (예측+갱신) */
#define SOP_CMD_EKF_P   0x67U   /* SOC EKF 1 스텝 (예측만) */
#define SOP_CMD_SOH     0x68U   /* SOH CNN 1 회 (시드 3 개 평균) */
#define SOP_CMD_SOH_Q   0x69U   /* SOH 완전 정수 경로 (SIMD) */
#define SOP_CMD_EKF_F64  0x6AU  /* 같은 구조 + double 산술 */
#define SOP_CMD_EKF2     0x6BU  /* 완전 2RC + float32 */
/* 아래 둘은 감사에서 추가.  SOP_BENCH_PACK 뒤에 둔다: 벤치 전용 명령이고,
 * 특히 PACK 은 셀별 상태 24 KB 를 들고 있어서 기본 빌드에 넣으면
 * build_size.csv 가 보고하는 배포 footprint 를 부풀린다.  배포 펌웨어는
 * 이 명령들을 싣지 않으므로 기본값은 off 다.
 *   make EXTRA_CFLAGS=-DSOP_BENCH_PACK MODEL_ID=pack_bench
 *  37.10 이 339.84 us 를 "단계별 최대의 합" 이라고
 * 다시 라벨했고, 37.21 이 그 합산 오차를 트림+반전 80 us 구간에서만 쟀다.
 * 네 단계를 한 DWT 창에서 도는 명령이 없었기 때문이다. */
#ifdef SOP_BENCH_PACK
#define SOP_CMD_CYCLE    0x70U  /* 특징+EKF+트림+반전 = 제어 1 주기 */
#define SOP_CMD_PACK     0x71U  /* 위를 N 셀에 돌리고 min 축약 */

/* 팩 규모.  dir 바이트의 상위 비트로 색인한다: dir = 방향 | (idx << 1).
 * 유선 형식(73 B)을 건드리지 않으려는 것이고, 방향은 0/1 뿐이라 자리가 남는다. */
#define SOP_PACK_NMAX 192U
static const uint16_t SOP_PACK_N[] = {1U, 12U, 48U, 96U, 192U};
#define SOP_PACK_NSET (sizeof(SOP_PACK_N) / sizeof(SOP_PACK_N[0]))
#endif  /* SOP_BENCH_PACK */
#define SOP_CMD_EKF2_F64 0x6CU  /* 완전 2RC + double */
#define SOP_CMD_FEAT_A8  0x6DU  /* 특징 갱신 1 샘플 — dR_fast 하나만 (채택) */

/* iters values that are status codes, not iteration counts.  0xFFFFFFFF is
 * already used for "body receive failed"; this one says the opcode itself
 * was not recognised, and `cycles` carries the opcode that was rejected. */
#define SOP_NACK_UNKNOWN_CMD 0xFFFFFFFEU
#define STACK_PATTERN 0xA5A5A5A5UL
/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/

UART_HandleTypeDef huart3;

/* USER CODE BEGIN PV */
static uint32_t *g_stack_paint_end;
/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
static void MX_GPIO_Init(void);
static void MX_USART3_UART_Init(void);
static void MX_ICACHE_Init(void);
/* USER CODE BEGIN PFP */
static void CEMA_DWT_Init(void);
static void Stack_Watermark_Init(void);
static uint32_t Stack_Highwater_Bytes(void);
/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */
extern uint32_t _sstack;
extern uint32_t _estack;

typedef struct __attribute__((packed))
{
  double voltage_v;
  double current_a;
  double temperature_c;
} CEMA_Raw_Sample;

static void UART_Send(const void *data, uint16_t size)
{
  (void)HAL_UART_Transmit(&huart3, (const uint8_t *)data, size, HAL_MAX_DELAY);
}

static void CEMA_DWT_Init(void)
{
  CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
#if defined(DWT_LAR)
  DWT->LAR = 0xC5ACCE55UL;
#endif
  DWT->CYCCNT = 0UL;
  DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
  __DSB();
  __ISB();
}

static void Stack_Watermark_Init(void)
{
  uintptr_t stack_pointer;
  uint32_t *cursor;

  __asm volatile ("mrs %0, msp" : "=r" (stack_pointer));
  if (stack_pointer > ((uintptr_t)&_sstack + 512U))
  {
    g_stack_paint_end = (uint32_t *)(stack_pointer - 256U);
    for (cursor = &_sstack; cursor < g_stack_paint_end; ++cursor)
    {
      *cursor = STACK_PATTERN;
    }
  }
  else
  {
    g_stack_paint_end = &_sstack;
  }
}

static uint32_t Stack_Highwater_Bytes(void)
{
  uint32_t *cursor;
  uint32_t *lowest_used = g_stack_paint_end;

  for (cursor = &_sstack; cursor < g_stack_paint_end; ++cursor)
  {
    if (*cursor != STACK_PATTERN)
    {
      lowest_used = cursor;
      break;
    }
  }
  return (uint32_t)((uintptr_t)&_estack - (uintptr_t)lowest_used);
}




/* ---- SOP 벤치 --------------------------------------------------------- */
/* 유선 형식은 dir(1 B) + float 6 개 + float 12 개 = 73 B, 정렬 없음.
 * Cortex-M33 은 packed 구조체의 float 를 직접 읽으면 UsageFault(UNALIGNED) 를
 * 낸다 — 실제로 냈다(CFSR 0x01000000, HFSR FORCED). 그래서 바이트로 받아
 * 정렬된 구조체로 memcpy 한다. */
#define SOP_WIRE_BYTES (1U + (6U + SOP_NFEAT) * 4U)

typedef struct
{
  uint8_t  dir;
  float    soc, soh, v_pre, v_limit, tau_s, current_a;
  float    x12[SOP_NFEAT];
} SOP_Request;

static void SOP_Unpack(const uint8_t *w, SOP_Request *q)
{
  q->dir = w[0];
  float f[6U + SOP_NFEAT];
  memcpy(f, w + 1, sizeof(f));
  q->soc = f[0]; q->soh = f[1]; q->v_pre = f[2];
  q->v_limit = f[3]; q->tau_s = f[4]; q->current_a = f[5];
  memcpy(q->x12, f + 6, sizeof(q->x12));
}

typedef struct
{
  uint32_t magic;
  uint32_t cycles;       /* DWT, 측정 구간만 */
  uint32_t iters;
  uint32_t stack_highwater_bytes;
  float    kf, ks, r_eff, i_star;
} SOP_Response;   /* 전부 4 B 정렬 -> packed 불필요, 크기 32 B */

static sop_feat_t g_feat;
static sop_ekf_t  g_ekf;
#ifdef SOP_BENCH_PACK
/* 팩 명령용 셀별 상태.  192 x (60 + 64) B = 23.8 KB, H563 의 SRAM 안에서
 * 넉넉하다.  하나를 공유해 돌리면 계산량은 재도 상태 메모리는 못 잰다. */
static sop_feat_t g_pack_feat[SOP_PACK_NMAX];
static sop_ekf_t  g_pack_ekf[SOP_PACK_NMAX];
#endif
static int        g_state_init;

static void SOH_Handle(int simd)
{
  static float x64[SOH_NIN];
  SOP_Response r = {0};
  uint32_t primask;
  r.magic = CEMA_PROTOCOL_MAGIC;
  if (HAL_UART_Receive(&huart3, (uint8_t *)x64, (uint16_t)sizeof(x64),
                       1000U) != HAL_OK)
  {
    r.iters = 0xFFFFFFFFU;
    UART_Send(&r, (uint16_t)sizeof(r));
    return;
  }
  primask = __get_PRIMASK();
  __disable_irq(); __DSB(); __ISB();
  DWT->CYCCNT = 0UL;
#if SOH_RIDGE
  (void)simd;
  float soh = soh_infer(x64);
#else
  float soh = simd ? soh_infer_simd(x64) : soh_infer(x64);
#endif
  r.cycles = DWT->CYCCNT;
  if (primask == 0U) { __enable_irq(); }
  r.i_star = soh;
  r.stack_highwater_bytes = Stack_Highwater_Bytes();
  UART_Send(&r, (uint16_t)sizeof(r));
}

static void SOP_Handle(uint8_t cmd)
{
  SOP_Request q;
  uint8_t wire[SOP_WIRE_BYTES];
  SOP_Response r = {0};
  uint32_t primask, it = 0U;
  float kf = 1.0f, ks = 1.0f, reff = 0.0f, ist = 0.0f;

  /* 타임아웃을 둔다.  호스트가 중간에 죽어도 보드가 대기 상태로 굳지 않는다 —
   * 굳으면 다음 QUERY 도 먹지 않아 매번 리셋해야 한다. */
  if (HAL_UART_Receive(&huart3, wire, (uint16_t)sizeof(wire),
                       500U) != HAL_OK)
  {
    SOP_Response e = {0};
    e.magic = CEMA_PROTOCOL_MAGIC;
    e.iters = 0xFFFFFFFFU;                 /* 수신 실패 표시 */
    e.stack_highwater_bytes = (uint32_t)sizeof(wire);
    UART_Send(&e, (uint16_t)sizeof(e));
    return;
  }
  SOP_Unpack(wire, &q);
  if (!g_state_init)
  {
    sop_feat_reset(&g_feat);
    sop_ekf_reset(&g_ekf, q.soc, q.soh);
    g_state_init = 1;
  }
  r.magic = CEMA_PROTOCOL_MAGIC;

#ifdef SOP_BENCH_PACK
  /* 팩 명령은 방향 바이트에 규모 색인을 얹어 보낸다. */
  uint8_t  dir_bits = (cmd == SOP_CMD_PACK) ? (uint8_t)(q.dir & 1U) : q.dir;
  uint16_t n_cells  = 1U;
  if (cmd == SOP_CMD_PACK)
  {
    uint8_t idx = (uint8_t)(q.dir >> 1);
    if (idx >= SOP_PACK_NSET) { idx = (uint8_t)(SOP_PACK_NSET - 1U); }
    n_cells = SOP_PACK_N[idx];
  }
#endif

  /* 트림은 REFF/SOLVE 측정 밖에서 미리 구한다 — 단계를 섞지 않기 위해.
   * CYCLE 과 PACK 은 정반대가 목적이므로 제외한다: 통합 주기를 재는 것이라
   * 트림이 창 안에 있어야 한다. */
  if (cmd != SOP_CMD_TRIM && cmd != SOP_CMD_FULL
#ifdef SOP_BENCH_PACK
      && cmd != SOP_CMD_CYCLE && cmd != SOP_CMD_PACK
#endif
     )
  {
    sop_trim((sop_dir_t)q.dir, q.x12, &kf, &ks);
  }

  primask = __get_PRIMASK();
  __disable_irq();
  __DSB(); __ISB();
  DWT->CYCCNT = 0UL;
  switch (cmd)
  {
#ifndef SOP_A8_ONLY
    case SOP_CMD_FEAT:
    {
      float xx[SOP_NFEAT];
      (void)sop_feat_update(&g_feat, 1.0f, q.current_a, q.v_pre, 25.0f,
                            q.soc, q.soh, xx);
      break;
    }
#endif
    case SOP_CMD_FEAT_A8:
    {
      /* 채택 구성.  x 는 하나뿐이지만 배열로 받아 호출 규약을 같게 둔다. */
      float x1[1];
      (void)sop_feat_update_a8(&g_feat, 1.0f, q.current_a, q.v_pre,
                               q.soc, q.soh, x1);
      break;
    }
    case SOP_CMD_EKF:
    case SOP_CMD_EKF_P:
      ist = sop_ekf_step(&g_ekf, 1.0f, q.current_a, q.v_pre, q.soh,
                         cmd == SOP_CMD_EKF);
      break;
    case SOP_CMD_EKF_F64:
      ist = sop_ekf_step_f64(&g_ekf, 1.0f, q.current_a, q.v_pre, q.soh, 1);
      break;
    case SOP_CMD_EKF2:
      ist = sop_ekf2_step(&g_ekf, 1.0f, q.current_a, q.v_pre, q.soh, 1);
      break;
    case SOP_CMD_EKF2_F64:
      ist = sop_ekf2_step_f64(&g_ekf, 1.0f, q.current_a, q.v_pre, q.soh, 1);
      break;
    case SOP_CMD_REFF:
      reff = sop_r_eff((sop_dir_t)q.dir, q.soc, q.soh, q.current_a, q.tau_s,
                       kf, ks);
      break;
    case SOP_CMD_TRIM:
      sop_trim((sop_dir_t)q.dir, q.x12, &kf, &ks);
      break;
    case SOP_CMD_SOLVE:
      ist = sop_solve((sop_dir_t)q.dir, q.soc, q.soh, q.v_pre, q.v_limit,
                      q.tau_s, kf, ks, &it);
      break;
#ifdef SOP_BENCH_PACK
    case SOP_CMD_CYCLE:
    {
      /* 배치에서 한 주기에 실제로 도는 것: 특징 갱신 -> SOC EKF ->
       * 트림 -> 반전.  단계별 측정의 합과 비교하려고 하나의 창에 둔다. */
      float x1[1];
      (void)sop_feat_update_a8(&g_feat, 1.0f, q.current_a, q.v_pre,
                               q.soc, q.soh, x1);
      (void)sop_ekf_step(&g_ekf, 1.0f, q.current_a, q.v_pre, q.soh, 1);
      sop_trim((sop_dir_t)q.dir, q.x12, &kf, &ks);
      ist = sop_solve((sop_dir_t)q.dir, q.soc, q.soh, q.v_pre, q.v_limit,
                      q.tau_s, kf, ks, &it);
      break;
    }
    case SOP_CMD_PACK:
    {
      /* 직렬 팩의 한 주기: 셀마다 같은 주기를 돌고 min 을 취한다.
       * 셀 상태를 공유하지 않고 셀당 하나씩 둔다 — 계산량뿐 아니라 상태
       * 메모리도 실제 규모로 재기 위해서다 (셀당 124 B). */
      float x1[1];
      float best = 1.0e30f;
      for (uint16_t c = 0U; c < n_cells; ++c)
      {
        float kfc = 1.0f, ksc = 1.0f;
        uint32_t itc = 0U;
        (void)sop_feat_update_a8(&g_pack_feat[c], 1.0f, q.current_a, q.v_pre,
                                 q.soc, q.soh, x1);
        (void)sop_ekf_step(&g_pack_ekf[c], 1.0f, q.current_a, q.v_pre,
                           q.soh, 1);
        sop_trim((sop_dir_t)dir_bits, q.x12, &kfc, &ksc);
        float ic = sop_solve((sop_dir_t)dir_bits, q.soc, q.soh, q.v_pre,
                             q.v_limit, q.tau_s, kfc, ksc, &itc);
        if (ic < best) { best = ic; kf = kfc; ks = ksc; it = itc; }
      }
      ist = best;
      break;
    }
#endif  /* SOP_BENCH_PACK */
    case SOP_CMD_FULL:
    default:
      sop_trim((sop_dir_t)q.dir, q.x12, &kf, &ks);
      ist = sop_solve((sop_dir_t)q.dir, q.soc, q.soh, q.v_pre, q.v_limit,
                      q.tau_s, kf, ks, &it);
      break;
  }
  r.cycles = DWT->CYCCNT;
  if (primask == 0U) { __enable_irq(); }

  r.iters = it;
  r.stack_highwater_bytes = Stack_Highwater_Bytes();
  r.kf = kf; r.ks = ks; r.r_eff = reff; r.i_star = ist;
  UART_Send(&r, (uint16_t)sizeof(r));
}

static void SOP_Send_Query(void)
{
  const uint32_t response[8] = {
      CEMA_PROTOCOL_MAGIC,
      CEMA_PROTOCOL_VERSION,
      SystemCoreClock,
      (uint32_t)SOP_WIRE_BYTES,
      (uint32_t)sizeof(SOP_Response),
      (uint32_t)(SOP_NS * SOP_NH * SOP_NRANK * 2 * 4 * 2),  /* ECM 격자 바이트 */
      (uint32_t)(SOP_NS * SOP_NH * 2 * 4),                  /* OCV 바이트 */
      (uint32_t)SOP_NFEAT};
  UART_Send(response, (uint16_t)sizeof(response));
}

static void CEMA_Protocol_Loop(void)
{
  uint8_t command;

  for (;;)
  {
    if (HAL_UART_Receive(&huart3, &command, 1U, HAL_MAX_DELAY) != HAL_OK)
    {
      continue;
    }
    if (command == SOP_CMD_QUERY)
    {
      SOP_Send_Query();
    }
    else if (command == SOP_CMD_SOH)
    {
      SOH_Handle(0);
    }
    else if (command == SOP_CMD_SOH_Q)
    {
#if SOH_RIDGE
      /* The ridge model has no integer path to compare against.  Say so
       * rather than returning the float timing under the integer opcode. */
      SOP_Response r = {0};
      r.magic  = CEMA_PROTOCOL_MAGIC;
      r.cycles = (uint32_t)SOP_CMD_SOH_Q;
      r.iters  = SOP_NACK_UNKNOWN_CMD;
      UART_Send(&r, (uint16_t)sizeof(r));
#else
      SOH_Handle(1);
#endif
    }
    else if (command == SOP_CMD_REFF || command == SOP_CMD_TRIM
             || command == SOP_CMD_SOLVE || command == SOP_CMD_FULL
             || command == SOP_CMD_FEAT_A8 || command == SOP_CMD_EKF
             || command == SOP_CMD_EKF_P || command == SOP_CMD_EKF_F64
             || command == SOP_CMD_EKF2 || command == SOP_CMD_EKF2_F64
#ifdef SOP_BENCH_PACK
             || command == SOP_CMD_CYCLE || command == SOP_CMD_PACK
#endif
#ifndef SOP_A8_ONLY
             || command == SOP_CMD_FEAT
#endif
             )
    {
      SOP_Handle(command);
    }
    else
    {
      /* An unknown command must be answered, not dropped.  The loop used to
       * fall through here in silence, so the 72-byte body that followed was
       * read back as a stream of command bytes; a valid byte inside it then
       * produced a plausible response to a command that was never handled.
       * sec 33.7 hit exactly this while probing whether FEAT had been
       * compiled out of the deployment build, and got 32 bytes back from a
       * binary that provably did not contain it.
       *
       * Drain the body, then NACK with the rejected opcode echoed back so a
       * host can tell "not supported" from "wrong answer". */
      uint8_t drain[SOP_WIRE_BYTES];
      SOP_Response nack = {0};
      (void)HAL_UART_Receive(&huart3, drain, (uint16_t)sizeof(drain), 500U);
      nack.magic = CEMA_PROTOCOL_MAGIC;
      nack.iters = SOP_NACK_UNKNOWN_CMD;
      nack.cycles = (uint32_t)command;
      nack.stack_highwater_bytes = (uint32_t)sizeof(drain);
      UART_Send(&nack, (uint16_t)sizeof(nack));
    }
  }
}

/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{

  /* USER CODE BEGIN 1 */

  /* USER CODE END 1 */

  /* MCU Configuration--------------------------------------------------------*/

  /* Reset of all peripherals, Initializes the Flash interface and the Systick. */
  HAL_Init();

  /* USER CODE BEGIN Init */

  /* USER CODE END Init */

  /* Configure the system clock */
  SystemClock_Config();

  /* USER CODE BEGIN SysInit */

  /* USER CODE END SysInit */

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  MX_USART3_UART_Init();
#ifndef SOP_NO_ICACHE
  MX_ICACHE_Init();
#endif  /* built with -DSOP_NO_ICACHE to test whether a timing difference
         * between two images is an instruction-cache placement effect */
  /* USER CODE BEGIN 2 */
  CEMA_DWT_Init();

  Stack_Watermark_Init();
  HAL_GPIO_WritePin(LED1_GPIO_Port, LED1_Pin, GPIO_PIN_SET);
  CEMA_Protocol_Loop();
  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
  while (1)
  {
    /* USER CODE END WHILE */

    /* USER CODE BEGIN 3 */
  }
  /* USER CODE END 3 */
}

/**
  * @brief System Clock Configuration
  * @retval None
  */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  /** Configure the main internal regulator output voltage
  */
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE0);

  while(!__HAL_PWR_GET_FLAG(PWR_FLAG_VOSRDY)) {}

  /** Initializes the RCC Oscillators according to the specified parameters
  * in the RCC_OscInitTypeDef structure.
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSE;
  RCC_OscInitStruct.HSEState = RCC_HSE_BYPASS_DIGITAL;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLL1_SOURCE_HSE;
  RCC_OscInitStruct.PLL.PLLM = 4;
  RCC_OscInitStruct.PLL.PLLN = 250;
  RCC_OscInitStruct.PLL.PLLP = 2;
  RCC_OscInitStruct.PLL.PLLQ = 2;
  RCC_OscInitStruct.PLL.PLLR = 2;
  RCC_OscInitStruct.PLL.PLLRGE = RCC_PLL1_VCIRANGE_1;
  RCC_OscInitStruct.PLL.PLLVCOSEL = RCC_PLL1_VCORANGE_WIDE;
  RCC_OscInitStruct.PLL.PLLFRACN = 0;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  /** Initializes the CPU, AHB and APB buses clocks
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2
                              |RCC_CLOCKTYPE_PCLK3;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV1;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV1;
  RCC_ClkInitStruct.APB3CLKDivider = RCC_HCLK_DIV1;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_5) != HAL_OK)
  {
    Error_Handler();
  }

  /** Configure the programming delay
  */
  __HAL_FLASH_SET_PROGRAM_DELAY(FLASH_PROGRAMMING_DELAY_2);
}

/**
  * @brief ICACHE Initialization Function
  * @param None
  * @retval None
  */
static void MX_ICACHE_Init(void)
{

  /* USER CODE BEGIN ICACHE_Init 0 */

  /* USER CODE END ICACHE_Init 0 */

  /* USER CODE BEGIN ICACHE_Init 1 */

  /* USER CODE END ICACHE_Init 1 */

  /** Enable instruction cache in 1-way (direct mapped cache)
  */
  if (HAL_ICACHE_ConfigAssociativityMode(ICACHE_1WAY) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_ICACHE_Enable() != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN ICACHE_Init 2 */

  /* USER CODE END ICACHE_Init 2 */

}

/**
  * @brief USART3 Initialization Function
  * @param None
  * @retval None
  */
static void MX_USART3_UART_Init(void)
{

  /* USER CODE BEGIN USART3_Init 0 */

  /* USER CODE END USART3_Init 0 */

  /* USER CODE BEGIN USART3_Init 1 */

  /* USER CODE END USART3_Init 1 */
  huart3.Instance = USART3;
  huart3.Init.BaudRate = 921600;
  huart3.Init.WordLength = UART_WORDLENGTH_8B;
  huart3.Init.StopBits = UART_STOPBITS_1;
  huart3.Init.Parity = UART_PARITY_NONE;
  huart3.Init.Mode = UART_MODE_TX_RX;
  huart3.Init.HwFlowCtl = UART_HWCONTROL_NONE;
  huart3.Init.OverSampling = UART_OVERSAMPLING_16;
  huart3.Init.OneBitSampling = UART_ONE_BIT_SAMPLE_DISABLE;
  huart3.Init.ClockPrescaler = UART_PRESCALER_DIV1;
  huart3.AdvancedInit.AdvFeatureInit = UART_ADVFEATURE_NO_INIT;
  if (HAL_UART_Init(&huart3) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_UARTEx_SetTxFifoThreshold(&huart3, UART_TXFIFO_THRESHOLD_1_8) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_UARTEx_SetRxFifoThreshold(&huart3, UART_RXFIFO_THRESHOLD_1_8) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_UARTEx_DisableFifoMode(&huart3) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN USART3_Init 2 */

  /* USER CODE END USART3_Init 2 */

}

/**
  * @brief GPIO Initialization Function
  * @param None
  * @retval None
  */
static void MX_GPIO_Init(void)
{
  GPIO_InitTypeDef GPIO_InitStruct = {0};
  /* USER CODE BEGIN MX_GPIO_Init_1 */

  /* USER CODE END MX_GPIO_Init_1 */

  /* GPIO Ports Clock Enable */
  __HAL_RCC_GPIOC_CLK_ENABLE();
  __HAL_RCC_GPIOH_CLK_ENABLE();
  __HAL_RCC_GPIOB_CLK_ENABLE();
  __HAL_RCC_GPIOD_CLK_ENABLE();

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(LED1_GPIO_Port, LED1_Pin, GPIO_PIN_RESET);

  /*Configure GPIO pin : USER_BUTTON_Pin */
  GPIO_InitStruct.Pin = USER_BUTTON_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_INPUT;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(USER_BUTTON_GPIO_Port, &GPIO_InitStruct);

  /*Configure GPIO pin : LED1_Pin */
  GPIO_InitStruct.Pin = LED1_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(LED1_GPIO_Port, &GPIO_InitStruct);

  /* USER CODE BEGIN MX_GPIO_Init_2 */

  /* USER CODE END MX_GPIO_Init_2 */
}

/* USER CODE BEGIN 4 */

/* USER CODE END 4 */

/**
  * @brief  This function is executed in case of error occurrence.
  * @param None
  * @retval None
  */
void Error_Handler(void)
{
  /* USER CODE BEGIN Error_Handler_Debug */
  /* User can add his own implementation to report the HAL error return state */
  __disable_irq();
  while (1)
  {
  }
  /* USER CODE END Error_Handler_Debug */
}
#ifdef USE_FULL_ASSERT
/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  * @retval None
  */
void assert_failed(uint8_t *file, uint32_t line)
{
  /* USER CODE BEGIN 6 */
  /* User can add his own implementation to report the file name and line number,
     ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */
