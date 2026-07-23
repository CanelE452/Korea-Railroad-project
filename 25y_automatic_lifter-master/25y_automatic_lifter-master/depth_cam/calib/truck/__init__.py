# calib/truck — Phase B (트럭 적재) 인지 모듈.
#
#   lasers.py        — TFmini-S 듀얼 레이저 리더 + 순수 로직 감지기
#                      (적재면 모서리 동시 급감 / 안착 해제 판정)
#   truck_adapter.py — SMOKE Detection → (ψ_truck, d_lateral, d_forward)
#   smoke_source.py  — truck_loading/ SMOKE 번들 lazy 래퍼 (torch 필요)
#
# 감지기/어댑터는 I/O 없는 순수 로직 — 하드웨어 없이 완전 단위테스트 가능.
