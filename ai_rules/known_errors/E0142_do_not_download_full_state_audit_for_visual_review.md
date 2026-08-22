# E0142: Do not download a full state audit for visual review

## Error

When the human asked to download ID189 so they could inspect it, the first transfer attempted the full Browser directory. That directory was about 40 GiB because 1862 per-turn NPZ files held every MCTS predicted state. The SSH path was severely degraded and only transferred about 15 MiB in more than an hour.

## Rule

Distinguish archival delivery from visual review before transfer. For visual review, create an immutable derived bundle containing HTML and real observation images while leaving raw NPZ and standalone JSON in the canonical verified archive. Report all exclusions explicitly.

Transfer one checksummed archive rather than thousands of files. If the ProxyJump link repeatedly stalls, stage the archive on the VPN host using resumable SFTP, then copy it over the local VPN link.

Never describe a view-only bundle as the complete audit archive.
