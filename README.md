# KR Market Momentum

## 한국어

### 개요
한국 주식 시장 데이터를 기반으로 유니버스를 구성하고, 종가 데이터를 다운로드한 뒤 랭크 모멘텃 전략을 빠르게 백테스트하는 유틸리티 스크립트 모음입니다.

### 구성 파일
- `1_build_universe2.py`
  - 시가총액/거래대금(Amount) 기준으로 유니버스 생성
  - Top N 또는 Top % 필터 지원
  - 스코프(KOSPI/KOSDAQ 등) 설정 가능

- `2_download_update_close.py`
  - 종가 데이터 다운로드/업데이트
  - YF 실패 시 FDR로 자동 보완 옵션

- `5_quick_backtest.py`
  - 랭크 모멘텃 전략 그리드 탐색
  - EQUAL / RP(공분산 기반 리스크패리티) 가중 지원
  - 상위 조합 성과 요약 및 벤치마크 비교 시각화

### 빠른 시작
1. 유니버스 생성
   - `1_build_universe2.py` 설정값 수정
   - 실행: `python 1_build_universe2.py`

2. 종가 다운로드
   - `2_download_update_close.py` 설정값 수정
   - 실행: `python 2_download_update_close.py`

3. 퀵 백테스트 실행
   - `5_quick_backtest.py` 설정값 수정
   - 실행: `python 5_quick_backtest.py`

### 참고
- 전략 파라미터 튜닝은 `5_quick_backtest.py`의 리스트/범위에서 조정
- RP 가중은 최근 N일 수익률 기반 공분산으로 계산

---

## English

### Overview
A set of utility scripts to build a Korean equity universe, download close prices, and run fast rank-momentum backtests.

### Files
- `1_build_universe2.py`
  - Build universe using market cap or amount filters
  - Supports Top N or Top % screening
  - Configurable scopes (KOSPI/KOSDAQ, etc.)

- `2_download_update_close.py`
  - Download/update close prices
  - Optional fallback to FDR when YF fails

- `5_quick_backtest.py`
  - Grid search for rank momentum
  - Supports EQUAL / RP (covariance-based risk parity) weights
  - Prints summary stats and plots against benchmarks

### Quick start
1. Build universe
   - Edit settings in `1_build_universe2.py`
   - Run: `python 1_build_universe2.py`

2. Download close prices
   - Edit settings in `2_download_update_close.py`
   - Run: `python 2_download_update_close.py`

3. Run quick backtest
   - Edit settings in `5_quick_backtest.py`
   - Run: `python 5_quick_backtest.py`

### Notes
- For parameter tuning, adjust lists/ranges in `5_quick_backtest.py`
- RP weights use covariance of recent returns
