# SFT1 真实输出检查

Checkpoint: epoch_001 / step 27. a100-1 CPU BF16 SDPA; first 4 heldout records; greedy decoding, max 512 new tokens. All 702 saved adapter tensors verified. All four responses ended at EOS after </think> and omitted the action block; 0/4 pass, none truncated. This is a four-sample diagnostic, not full validation or rollout success measurement.

## vagen-step60/000009

Tokens: 70; stop: eos; format: fail (missing action block).

```text
<think><observation>There is a green lettuce on the counter to the right of the sink.</observation><reasoning>I need to move towards the lettuce. The lettuce is to the right of the sink, so I should move rightward.</reasoning><prediction>After moving right, the lettuce should be more visible.</prediction></think><|im_end|>
```

## vagen-step60/000019

Tokens: 82; stop: eos; format: fail (missing action block).

```text
<think><observation>There is a GarbageCan on the right side of the room, near the wall.</observation><reasoning>I should move towards the GarbageCan to get closer to it. The GarbageCan is to the right, so I should move rightward.</reasoning><prediction>After moving right, the GarbageCan should become more visible and closer.</prediction></think><|im_end|>
```

## vagen-step60/000029

Tokens: 54; stop: eos; format: fail (missing action block).

```text
<think><observation>There is a laptop on the table to the right.</observation><reasoning>I should move towards the laptop to get closer to it.</reasoning><prediction>After moving, the laptop should be more visible and closer.</prediction></think><|im_end|>
```

## vagen-step60/000039

Tokens: 83; stop: eos; format: fail (missing action block).

```text
<think><observation>There is a table with a lamp on it to the left and a dresser with an alarm clock on top to the right.</observation><reasoning>I should move towards the dresser to find the alarm clock. The alarm clock is likely on the right side of the room.</reasoning><prediction>After moving right, the alarm clock should be visible on the dresser.</prediction></think><|im_end|>
```
