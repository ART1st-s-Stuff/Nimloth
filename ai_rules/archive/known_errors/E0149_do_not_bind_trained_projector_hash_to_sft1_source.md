# E0149 — Do not bind a trained projector hash to the SFT1 source

## Error

ID191 attempt0 pinned the ID74 trained `state_proj.pt` SHA256
`e789a672...d4063` as if it identified the SFT1 `slot_projector.pt`. The real
SFT1 source SHA256 is `340d90a8...42ce0`, so Job `529701` failed preflight.

## Rule

When an experiment distinguishes a frozen SFT1 projector from a later trained
SFT2/ID74 projector, hash the exact path named by the contract. Never transfer
a digest from a semantic summary or derivative checkpoint merely because both
files instantiate `SharedSlotProjector`. Retain the full checkpoint-identity
comparison against the authoritative source result in addition to shell pins.
