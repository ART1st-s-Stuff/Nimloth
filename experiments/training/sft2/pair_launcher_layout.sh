#!/bin/bash

pair_layout_values() {
  local mode=$1 host=$2 procid=$3
  if [[ "$mode" == one_rank_per_node ]]; then
    printf '%s 1\n' "$procid"
    return
  fi
  case "$host" in
    dgx-27*) printf '0 3\n' ;;
    dgx-54*) printf '3 1\n' ;;
    *) echo "unexpected host $host" >&2; return 2 ;;
  esac
}
