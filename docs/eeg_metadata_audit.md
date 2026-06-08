# EEG And Clinical Source Metadata Audit

This audit was generated from the current external data paths configured in `configs/paths.example.yaml`. It records non-identifying metadata only.

## Supervised Baseline EEG Summary

- Records audited: 38 baseline EO/EC records from supervised patients.
- Sampling-rate values: 250.0 Hz.
- Channel-count values: 62.
- Trial-count values: 1.
- Recording duration: mean 188.42 s, range 101.04-247.76 s.
- Sample points per file: range 25259-61940.

## EEG Metadata Tables

- Record-level audit: `results/tables/eeg_recording_metadata_audit.csv`.
- Grouped summary: `results/tables/eeg_recording_summary.csv`.

## Clinical Workbook Structure

| Workbook | Sheet | Rows | Columns | Header candidates | Date/timing hits | Stroke/diagnosis hits | Ethics/consent hits | EEG/protocol hits |
|---|---|---:|---:|---|---:|---:|---:|---:|
| integrity_workbook | Sheet1 | 20 | 21 | 患者ID / 姓名 / FMA变化量 / 基线完整性 / 四阶段整体完整性 / 基线_睁眼 / 基线_闭眼 / 基线_运动想象 / 基线_抓握任务 / 即时_睁眼 / 即时_闭眼 / 即时_运动想象 / 即时_抓握任务 / 阶段_睁眼 / 阶段_闭眼 / 阶段_运动想象 / 阶段_抓握任务 / 最终_睁眼 / 最终_闭眼 / 最终_运动想象 / 最终_抓握任务 | 0 | 0 | 0 | 0 |
| clinical_workbook | Sheet1 | 51 | 13 | 编号 / 姓名 / 年龄 / 病程 / 性别 / 患病侧（手） / 治疗前FMA / 治疗后FMA / 治疗前MBI / 治疗后MBI / 缺少数据 / 脱落原因 / 核磁次数 | 1 | 0 | 0 | 10 |

## Interpretation

- The EEG metadata support reporting 250 Hz sampling, 62 retained channels, continuous single-trial files, and approximate baseline recording duration.
- The workbook structure audit found whether candidate timing, diagnosis, ethics/consent, or protocol terms appear in the source sheets, but it does not expose identifiable patient-level values.
- The current manuscript feature scripts start from the provided preprocessed `.set/.fdt` files and do not perform additional temporal filtering, artifact rejection, channel interpolation, or bad-channel removal after loading.
- Ethics approval number, consent wording, acquisition hardware, online reference scheme, raw preprocessing filters, artifact rejection, and exact recruitment criteria remain author-supplied items unless confirmed in a separate protocol document.
