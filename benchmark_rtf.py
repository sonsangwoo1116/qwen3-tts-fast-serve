#!/usr/bin/env python3
"""
Benchmark RTF (Real-Time Factor) for Qwen3-TTS batched server.

Sends N concurrent streaming requests and measures wall-clock time vs audio duration.
RTF = wall_time / audio_duration  (< 1.0 means real-time)
"""

import argparse
import io
import struct
import sys
import time
import threading
import wave

import requests

API_URL = "http://localhost:8003/v1/audio/speech"

# Korean test sentences of varying lengths
TEST_TEXTS = [
    "안녕하세요. 오늘 날씨가 참 좋습니다. 산책하기 딱 좋은 날이네요.",
    "고객님께서 문의하신 내용에 대해 안내드리겠습니다. 잠시만 기다려 주세요.",
    "지금부터 회의를 시작하겠습니다. 오늘 안건은 세 가지입니다.",
    "주문하신 상품은 내일 오전 중으로 배송될 예정입니다. 감사합니다.",
    "이번 달 매출이 전월 대비 십오 퍼센트 증가했습니다. 좋은 성과입니다.",
    "예약 확인 도와드리겠습니다. 성함과 연락처를 말씀해 주세요.",
    "오늘 점심 메뉴는 김치찌개와 된장찌개 중에서 골라주세요.",
    "내일 오전 열 시에 미팅이 잡혀 있습니다. 준비해 주세요.",
    "결제가 정상적으로 완료되었습니다. 영수증을 보내드리겠습니다.",
    "이 제품은 삼 년간 무상 보증이 제공됩니다. 편하게 사용하세요.",
    "서울에서 부산까지 케이티엑스로 약 두 시간 반 소요됩니다.",
    "다음 주 월요일까지 보고서를 제출해 주시기 바랍니다.",
    "비밀번호를 변경하시려면 마이페이지에서 설정을 변경해 주세요.",
    "오늘 저녁에 가족 모임이 있어서 일찍 퇴근하려고 합니다.",
    "신규 회원 가입 시 첫 달 무료 혜택이 제공됩니다.",
    "택배가 도착하면 문자로 알려드리겠습니다. 확인 부탁드립니다.",
    "이번 주말에 특별 할인 행사를 진행합니다. 많은 관심 부탁드립니다.",
    "시스템 점검으로 인해 오늘 밤 열두 시부터 새벽 두 시까지 서비스가 중단됩니다.",
    "문의사항이 있으시면 고객센터로 전화 주시거나 챗봇을 이용해 주세요.",
    "감사합니다. 좋은 하루 보내세요. 다음에 또 방문해 주세요.",
]

SAMPLE_RATE = 24000


def get_audio_duration_from_stream(resp: requests.Response) -> float:
    """Read streaming WAV response and return audio duration in seconds."""
    data = resp.content
    # Skip 44-byte WAV header, count PCM samples (16-bit mono)
    pcm_data = data[44:]
    num_samples = len(pcm_data) // 2  # 16-bit = 2 bytes per sample
    return num_samples / SAMPLE_RATE


def send_request(idx: int, text: str, results: list, voice: str = "Vivian", stream: bool = True):
    """Send a TTS request and record timing."""
    t0 = time.time()
    first_byte_time = None
    try:
        resp = requests.post(
            API_URL,
            json={
                "input": text,
                "voice": voice,
                "language": "Korean",
                "stream": stream,
            },
            stream=stream,
            timeout=120,
        )
        resp.raise_for_status()

        if stream:
            chunks = []
            for chunk in resp.iter_content(chunk_size=4096):
                if first_byte_time is None:
                    first_byte_time = time.time()
                chunks.append(chunk)
            total_data = b"".join(chunks)
        else:
            first_byte_time = time.time()
            total_data = resp.content

        wall_time = time.time() - t0
        ttfb = (first_byte_time - t0) if first_byte_time else wall_time

        # Calculate audio duration from total PCM data
        pcm_data = total_data[44:]  # skip WAV header
        num_samples = len(pcm_data) // 2
        audio_dur = num_samples / SAMPLE_RATE

        results[idx] = {
            "wall_time": wall_time,
            "ttfb": ttfb,
            "audio_dur": audio_dur,
            "text_len": len(text),
            "ok": True,
        }
    except Exception as e:
        wall_time = time.time() - t0
        results[idx] = {
            "wall_time": wall_time,
            "ttfb": 0,
            "audio_dur": 0,
            "text_len": len(text),
            "ok": False,
            "error": str(e),
        }


def run_benchmark(n_channels: int, voice: str = "Vivian", stream: bool = True):
    """Run N concurrent requests and compute RTF."""
    texts = [TEST_TEXTS[i % len(TEST_TEXTS)] for i in range(n_channels)]
    results = [None] * n_channels
    threads = []

    t_start = time.time()
    for i in range(n_channels):
        t = threading.Thread(target=send_request, args=(i, texts[i], results, voice, stream))
        threads.append(t)
        t.start()

    for t in threads:
        t.join(timeout=120)
    t_total = time.time() - t_start

    # Analyze results
    ok_results = [r for r in results if r and r["ok"]]
    fail_count = n_channels - len(ok_results)

    if not ok_results:
        print(f"  {n_channels:3d}ch: ALL FAILED")
        return None

    wall_times = [r["wall_time"] for r in ok_results]
    ttfbs = [r["ttfb"] for r in ok_results]
    audio_durs = [r["audio_dur"] for r in ok_results]

    max_wall = max(wall_times)
    avg_wall = sum(wall_times) / len(wall_times)
    avg_ttfb = sum(ttfbs) / len(ttfbs)
    avg_audio = sum(audio_durs) / len(audio_durs)
    total_audio = sum(audio_durs)

    # RTF: max wall time / avg audio duration
    # (all requests run concurrently, so RTF = slowest_request / its_audio_duration)
    rtfs = [w / a if a > 0 else 999 for w, a in zip(wall_times, audio_durs)]
    max_rtf = max(rtfs)
    avg_rtf = sum(rtfs) / len(rtfs)

    status = "PASS" if max_rtf < 1.0 else "FAIL"

    print(f"  {n_channels:3d}ch: RTF={max_rtf:.3f} (avg={avg_rtf:.3f}) | "
          f"wall={max_wall:.2f}s avg={avg_wall:.2f}s | "
          f"TTFB={avg_ttfb:.2f}s | "
          f"audio={avg_audio:.2f}s | "
          f"fail={fail_count} | {status}")

    return {
        "channels": n_channels,
        "max_rtf": max_rtf,
        "avg_rtf": avg_rtf,
        "max_wall": max_wall,
        "avg_wall": avg_wall,
        "avg_ttfb": avg_ttfb,
        "avg_audio": avg_audio,
        "total_audio": total_audio,
        "fail_count": fail_count,
        "status": status,
    }


def main():
    parser = argparse.ArgumentParser(description="TTS RTF Benchmark")
    parser.add_argument("--channels", type=str, default="1,5,10,15,20",
                        help="Comma-separated channel counts")
    parser.add_argument("--voice", type=str, default="Vivian")
    parser.add_argument("--rounds", type=int, default=2,
                        help="Number of rounds per channel count")
    parser.add_argument("--warmup", action="store_true", default=True,
                        help="Send warmup request first")
    parser.add_argument("--no-stream", action="store_true", default=False,
                        help="Use non-streaming mode")
    args = parser.parse_args()

    channels = [int(c) for c in args.channels.split(",")]
    use_stream = not args.no_stream

    print(f"=== TTS RTF Benchmark ===")
    print(f"Voice: {args.voice}, Rounds: {args.rounds}, Stream: {use_stream}")
    print(f"Channels: {channels}")
    print()

    # Warmup
    if args.warmup:
        print("Warming up...")
        results = [None]
        send_request(0, "안녕하세요. 테스트입니다.", results, args.voice, use_stream)
        if results[0] and results[0]["ok"]:
            print(f"  Warmup OK ({results[0]['wall_time']:.2f}s)\n")
        else:
            print(f"  Warmup FAILED: {results[0]}\n")
            return

    all_results = []
    for ch in channels:
        print(f"--- {ch} channels ---")
        for r in range(args.rounds):
            result = run_benchmark(ch, args.voice, use_stream)
            if result:
                all_results.append(result)
            # Brief pause between rounds
            time.sleep(1)
        print()

    # Summary table
    print("=" * 70)
    print(f"{'Ch':>4} | {'Best RTF':>9} | {'Avg RTF':>9} | {'Avg Wall':>9} | {'TTFB':>6} | Status")
    print("-" * 70)
    # Group by channel count, show best round
    from collections import defaultdict
    by_ch = defaultdict(list)
    for r in all_results:
        by_ch[r["channels"]].append(r)
    for ch in channels:
        if ch in by_ch:
            best = min(by_ch[ch], key=lambda x: x["max_rtf"])
            avg_rtf = sum(r["avg_rtf"] for r in by_ch[ch]) / len(by_ch[ch])
            print(f"{ch:4d} | {best['max_rtf']:9.3f} | {avg_rtf:9.3f} | "
                  f"{best['avg_wall']:8.2f}s | {best['avg_ttfb']:5.2f}s | {best['status']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
