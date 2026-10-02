#!/bin/sh
# sysinfo.sh - the machine facts every result needs. Writes markdown to stdout:
#   bench/sysinfo.sh > bench/results/<date>-<gpu>-<cpu>/MACHINE.md
echo "# Machine"
echo
echo "- Date: $(date -u +%Y-%m-%dT%H:%MZ)"
echo "- OS: $(. /etc/os-release 2>/dev/null; echo "$PRETTY_NAME"), kernel $(uname -r)"
echo "- CPU: $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | sed 's/^ //'), $(nproc) threads"
echo "- RAM: $(free -g | awk '/Mem:/{print $2}') GiB total"
if command -v dmidecode >/dev/null 2>&1 && sudo -n true 2>/dev/null; then
  echo "- RAM modules: $(sudo dmidecode -t memory | grep -E '^\s+Type: DDR' | sort | uniq -c | sed 's/^ *//;s/\s\+/ /g' | tr '\n' ' ')at $(sudo dmidecode -t memory | grep -m1 'Configured Memory Speed' | cut -d: -f2 | sed 's/^ //')"
fi
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.total,driver_version,pcie.link.gen.max,pcie.link.width.max --format=csv,noheader |
    while IFS=, read -r i n m d g w; do echo "- GPU $i:$n,$m, driver$d, PCIe gen$g x$w"; done
fi
command -v nvcc >/dev/null 2>&1 && echo "- CUDA toolkit: $(nvcc --version | grep -o 'release [0-9.]*')"
