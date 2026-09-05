# Legacy approval runtime只读快照（2026-09-05）

## Boundaries

- Source：`/workspace/remote2/nimloth/.trellis/.runtime/execution`
- Snapshot：`/workspace/remote2/nimloth/.local/audit/legacy-trellis-approval-runtime/20260905T065407Z`
- Source保持原样；未删除、改写、chmod或停止任何live runtime。
- Snapshot目录mode `0550`；8个payload、`manifest.json`、`manifest.sha256`均为`0440`。

## Verification

- 文件数：8 source JSON = 8 snapshot payloads。
- 每个文件在读取前后校验device/inode/size/mtime未变化；写入后及seal前再次比较source/snapshot SHA-256和bytes。
- Manifest SHA-256：`81ff75a98a6a7e9ee15256eef0caf3d13fd46fa1df6796333a42faf2ad667f16`。
- Aggregate：212 requests、202 receipts、401 assignments。
- Request status：201 approve、1 comment、2 pending、8 system_cancelled。

| File | Bytes | SHA-256 |
|---|---:|---|
| `pi_01a05646-e483-7e57-8500-918296664d95.json` | 67,670 | `7ff0e5d5efcea37ffbdeec5b8479e153878db1d3697844388ff2178311ff926f` |
| `pi_01a0565b-12e5-76d6-9c3b-e624a4a39191.json` | 7,417 | `350ffc1e04b016a3e3498c37a91a792d7890391bfbb3bd221be835aca0529552` |
| `pi_01a0574c-979e-76e3-aa35-25990a6514b2.json` | 456,765 | `3d569084354eaa4b56f45abdf413e366be9b23e2ad9143af913ceb7268c137ef` |
| `pi_01a057c7-a866-753d-a32f-706aa4a6725e.json` | 1,731 | `364f464623d60eb4b562d8ab4369f6a7777d29a0b3f38661d8a2b774c48e90d2` |
| `pi_01a057f0-f15e-7fe2-8c98-52e0fb714c05.json` | 87,968 | `382c240f2c32a3f65af045af8e8101547bc8a22355832e29e77f40e6e2d98527` |
| `pi_01a05ae2-6e3e-78a1-84f0-26d7c0b6822e.json` | 602,311 | `92c323e8eb1659d4ca0f03fcf9377c922f76df1b06306cd66e10bcaa58893460` |
| `pi_01a05c8d-6cb5-7ffc-b5ea-223e53834d29.json` | 68,771 | `ba76c0246f618b282a5f434a0d78728cd9599df6464bfdf0d3e7ebc18bf1bc0e` |
| `pi_01a06259-0bb6-7332-aad3-7b7ed77fceac.json` | 276,698 | `6bab2016d22a73c672164a0d7b546df5651fa0c90919e007a61e2b78bbf06877` |

Live runtime删除仍是独立破坏门禁；当前未获授权，保持blocked且不阻碍新架构验证。
