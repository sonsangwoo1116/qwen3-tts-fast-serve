#!/usr/bin/env python3
"""
Concurrent TTS streaming benchmark.
Measures RTF (Real-Time Factor) at various concurrency levels.

RTF < 1.0 = real-time capable
"""

import asyncio
import json
import time
import struct
import sys
import wave
import io
from urllib.request import Request, urlopen
from concurrent.futures import ThreadPoolExecutor

SERVER_URL = "http://localhost:8004/v1/audio/speech"
VOICE = "psb_voice"
SAMPLE_RATE = 24000

# Long Korean test sentences (60+ chars each, ~10s audio) for realistic streaming benchmark
TEST_SENTENCES = [
    "고객님께서 문의하신 내용에 대해 확인해 드리겠습니다. 먼저 본인 확인을 위해 가입 시 등록하신 휴대전화 번호를 말씀해 주시겠습니까?",
    "현재 고객님의 계좌에서 지난 삼월 십오일에 삼십만원이 자동이체로 출금되었으며, 잔액은 백이십삼만 사천오백원입니다.",
    "배송 상태를 조회해 드린 결과, 고객님의 택배는 현재 서울 송파구 배송센터에서 배송 출발하였으며 오늘 오후 다섯시까지 도착 예정입니다.",
    "가입하신 무제한 데이터 요금제는 월 오만구천원이며, 이번 달 사용량은 삼십이 기가바이트입니다. 요금제 변경을 원하시면 일번을 눌러주세요.",
    "지난 주 접수하신 서비스 수리 건에 대해 안내 드립니다. 현재 부품 교체가 완료되어 품질 검사 중이며, 내일 오전 중으로 수리 완료 예정입니다.",
    "오늘 예약하신 진료 일정을 확인해 드리겠습니다. 오후 두시 삼십분에 내과 김영수 전문의 선생님 진료가 예약되어 있습니다.",
    "고객님의 보험 계약 내용을 안내해 드리겠습니다. 현재 가입하신 상품은 종합건강보험이며, 월 보험료는 십이만 삼천원이고 만기일은 이천이십팔년 삼월입니다.",
    "죄송합니다만, 현재 시스템 점검으로 인해 일부 서비스 이용이 제한되고 있습니다. 점검은 오늘 밤 열한시까지 진행될 예정이오니 양해 부탁드립니다.",
    "주문하신 상품의 환불 처리가 완료되었습니다. 결제하신 신용카드로 칠만 팔천원이 환불되며, 카드사에 따라 영업일 기준 삼일에서 칠일 이내 반영됩니다.",
    "이번 달 프로모션 안내 드립니다. 신규 가입 고객에게는 첫 세 달간 월 요금의 삼십 퍼센트 할인이 적용되며, 추가로 포인트 만점이 지급됩니다.",
    "교통 정보를 안내해 드리겠습니다. 현재 강남역에서 서울역까지 지하철 이호선으로 약 이십오분 소요되며, 버스를 이용하시면 약 사십분 정도 걸립니다.",
    "전화 상담을 원하시면 일번, 문자 상담을 원하시면 이번, 채팅 상담을 원하시면 삼번을 눌러주세요. 이전 메뉴로 돌아가시려면 별표를 눌러주세요.",
    "지난달 결제 내역을 확인해 드리겠습니다. 총 결제 금액은 이백삼십사만 오천원이며, 가장 큰 결제 건은 삼월 이십일 백화점에서 구십팔만원입니다.",
    "항공편 예약이 확인되었습니다. 대한항공 케이이 팔공일편으로 삼월 이십오일 오전 아홉시 인천 출발, 도쿄 나리타 도착 현지 시간 열한시 삼십분입니다.",
    "고객님, 말씀하신 인터넷 연결 문제에 대해 원격 점검을 진행하겠습니다. 모뎀의 전원을 끄고 삼십초 후에 다시 켜주시면 자동으로 연결 상태를 확인하겠습니다.",
    "오늘 주문하신 음식은 현재 조리 중이며, 예상 배달 시간은 약 삼십오분입니다. 배달 기사님이 출발하시면 실시간으로 위치 추적이 가능합니다.",
    "연말정산 관련 안내 드립니다. 올해 소득공제 서류 제출 마감일은 이월 이십팔일이며, 필요 서류는 홈페이지에서 다운로드하실 수 있습니다.",
    "자동차 정기 검사 안내 드립니다. 고객님의 차량 정기 검사 만료일이 다음 달 십오일입니다. 가까운 검사소를 안내해 드릴까요?",
    "회원 등급 안내 드립니다. 고객님은 현재 골드 등급이시며, 다음 달 실적에 따라 플래티넘 등급으로 승급 가능합니다. 승급 조건은 월 구매 금액 오십만원 이상입니다.",
    "감사합니다. 상담이 도움이 되셨다면 통화 종료 후 만족도 평가에 참여해 주시면 서비스 개선에 큰 도움이 됩니다. 좋은 하루 보내세요.",
]


def send_request(sentence_idx: int) -> dict:
    """Send a single TTS request and measure timing."""
    text = TEST_SENTENCES[sentence_idx % len(TEST_SENTENCES)]
    payload = json.dumps({"input": text, "voice": VOICE, "stream": True}).encode()
    req = Request(SERVER_URL, data=payload, headers={"Content-Type": "application/json"})

    t0 = time.monotonic()
    first_byte_time = None
    chunks = []

    resp = urlopen(req, timeout=120)
    while True:
        chunk = resp.read(4096)
        if not chunk:
            break
        if first_byte_time is None:
            first_byte_time = time.monotonic()
        chunks.append(chunk)

    t1 = time.monotonic()
    data = b"".join(chunks)

    # Parse WAV to get audio duration
    try:
        with wave.open(io.BytesIO(data), "rb") as wf:
            frames = wf.getnframes()
            sr = wf.getframerate()
            audio_dur = frames / sr
    except Exception:
        # Fallback: estimate from raw size (16-bit mono PCM with 44-byte header)
        audio_dur = max(0, len(data) - 44) / (SAMPLE_RATE * 2)

    wall = t1 - t0
    ttfb = (first_byte_time - t0) if first_byte_time else wall
    rtf = wall / audio_dur if audio_dur > 0 else float("inf")

    return {
        "idx": sentence_idx,
        "chars": len(text),
        "audio_dur": audio_dur,
        "wall": wall,
        "ttfb": ttfb,
        "rtf": rtf,
    }


async def run_concurrent(n_channels: int, executor: ThreadPoolExecutor) -> list:
    """Run n_channels concurrent TTS requests."""
    loop = asyncio.get_event_loop()
    tasks = [
        loop.run_in_executor(executor, send_request, i)
        for i in range(n_channels)
    ]
    return await asyncio.gather(*tasks, return_exceptions=True)


def print_results(n_channels: int, results: list):
    ok = [r for r in results if isinstance(r, dict)]
    errors = [r for r in results if not isinstance(r, dict)]

    if not ok:
        print(f"  ALL {len(errors)} FAILED")
        return

    rtfs = [r["rtf"] for r in ok]
    walls = [r["wall"] for r in ok]
    ttfbs = [r["ttfb"] for r in ok]
    audio_durs = [r["audio_dur"] for r in ok]

    avg_rtf = sum(rtfs) / len(rtfs)
    max_rtf = max(rtfs)
    avg_wall = sum(walls) / len(walls)
    max_wall = max(walls)
    avg_ttfb = sum(ttfbs) / len(ttfbs)
    avg_audio = sum(audio_durs) / len(audio_durs)
    realtime = sum(1 for r in rtfs if r < 1.0)

    status = "OK" if avg_rtf < 1.0 else "SLOW"
    print(f"  {n_channels:>3} ch | RTF avg={avg_rtf:.3f} max={max_rtf:.3f} | "
          f"wall avg={avg_wall:.2f}s max={max_wall:.2f}s | "
          f"TTFB avg={avg_ttfb:.2f}s | "
          f"audio avg={avg_audio:.2f}s | "
          f"realtime={realtime}/{len(ok)} | "
          f"err={len(errors)} | [{status}]")

    return avg_rtf


def main():
    channel_counts = [1, 5, 10, 15, 20]
    n_rounds = 2
    if len(sys.argv) > 1:
        channel_counts = [int(x) for x in sys.argv[1].split(",")]
    if len(sys.argv) > 2:
        n_rounds = int(sys.argv[2])

    print("=" * 100)
    print(f"Concurrent TTS Benchmark — server={SERVER_URL} voice={VOICE}")
    print(f"Channel counts: {channel_counts}, rounds={n_rounds}")
    print("=" * 100)

    # Warmup
    print("\nWarming up (2 requests)...")
    for i in range(2):
        r = send_request(i)
        print(f"  warmup {i+1}: wall={r['wall']:.2f}s rtf={r['rtf']:.3f} audio={r['audio_dur']:.2f}s")

    print("\nBenchmark results:")
    print("-" * 100)

    results_summary = {}
    for n in channel_counts:
        round_rtfs = []
        for rd in range(1, n_rounds + 1):
            print(f"  [Round {rd}/{n_rounds}]")
            executor = ThreadPoolExecutor(max_workers=n)
            results = asyncio.run(run_concurrent(n, executor))
            r = print_results(n, results)
            if r is not None:
                round_rtfs.append(r)
            executor.shutdown(wait=False)
            time.sleep(1)  # brief pause between rounds
        if round_rtfs:
            avg = sum(round_rtfs) / len(round_rtfs)
            results_summary[n] = avg
            print(f"  >>> {n:>3} ch avg over {len(round_rtfs)} rounds: RTF={avg:.3f}")
        print()

    print("-" * 100)
    print("\nSummary:")
    for n, rtf in results_summary.items():
        mark = "PASS" if rtf < 1.0 else "FAIL"
        print(f"  {n:>3} channels: RTF={rtf:.3f} [{mark}]")

    print()


if __name__ == "__main__":
    main()
