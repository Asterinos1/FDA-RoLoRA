import os, re

with open('FDA_vs_Baselines_v2.md', 'r', encoding='utf-8') as f:
    text = f.read()

print(f"Total file size: {len(text):,} characters")

# Check metadata
for line in text.splitlines()[:25]:
    if any(k in line for k in ["Total Runs", "Unique", "Model"]):
        print("Metadata:", line)

# Check intro string for config count
for line in text.splitlines():
    if "across all" in line and "parameter configurations" in line:
        print("Intro config count:", line[:120], "...")

# Check table 1 configs
configs_t1 = []
for line in text.splitlines():
    if line.startswith("| **") and ("Non-IID" in line or "alpha=1000" in line):
        m = re.search(r'\*\*([^*]+)\*\*', line)
        if m and m.group(1) not in configs_t1:
            configs_t1.append(m.group(1))

print(f"\nUnique configuration groups in Table 1 & 2 ({len(configs_t1)} total):")
for i, c in enumerate(configs_t1, 1):
    print(f"  {i:02d}. {c}")

# Check Section 4.9 table benchmarks
configs_t49 = []
for line in text.splitlines():
    if line.startswith("| **") and ("alpha=0.5" in line):
        m = re.search(r'\*\*([^*]+)\*\*', line)
        if m and m.group(1) not in configs_t49:
            configs_t49.append(m.group(1))

print(f"\nUnique benchmark groups in Section 4.9 Table ({len(configs_t49)} total):")
for i, c in enumerate(configs_t49, 1):
    print(f"  {i}. {c}")

# Check key findings mentioned
findings = [
    "SST-2 ($C=50$): Decisive Prevention of Majority-Class Collapse",
    "+25.86%",
    "QQP ($C=20$): Outperforming RoLoRA on Semantic Paraphrasing",
    "MNLI ($C=20$): Multi-Seed Replicated Prevention of RoLoRA E20 Collapse",
    "Factor Matrix Symmetry Telemetry Across All 8 Non-IID Benchmarks",
]
print("\nKey findings verification:")
for finding in findings:
    print(f"  [{'PASS' if finding in text else 'FAIL'}] {finding}")
