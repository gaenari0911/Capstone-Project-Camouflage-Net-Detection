# Third-party code, model, and data notices

This is a research-code release, not a claim that every component or dataset is MIT-licensed or approved for commercial redistribution. The existing root [LICENSE](LICENSE) is preserved. It does not replace licenses and notices carried by third-party or derived components; obligations depend on the component and how it is combined/distributed.

## Custom Ultralytics / DualHeadSegment

The actual local custom package is included at `third_party/ultralytics_custom/`, with its original **GNU AGPL v3** [LICENSE](third_party/ultralytics_custom/LICENSE), source headers, package metadata and attribution preserved. `models/yolov8-seg-custom.yaml` also retains its AGPL source header. Do not describe that code as MIT-only or remove its notices.

Historical project model branch: [feature/Custom_Model, commit90bdaf0](https://github.com/gaenari0911/Capstone-Project-Camouflage-Net-Detection/tree/90bdaf0714b39d71447691d303a19af808a95ee6). The public export's per-file hashes in `results/source_manifest.json` identify the local files actually used; identical content with every historical branch file is not asserted. Upstream: [Ultralytics license](https://github.com/ultralytics/ultralytics/blob/main/LICENSE).

Only package source/configuration and packaging metadata were exported; bundled sample photographs, weights, caches, virtual environments, documentation site builds and upstream CI were excluded. Additional source changes in the original research workflow should be reviewed through the exported hashes. No blanket commercial-use clearance is given for the combined pipeline.

## AnyDoor and its bundled components

The experiment used official AnyDoor revision `44ca2b2a70ec2cf107f3d26a5b46def6670fb0a5` plus the project compatibility/shape-control adapters. The original AnyDoor source, generator weights and generated image assets are **not** redistributed here.

- [Pinned AnyDoor source license](https://github.com/ali-vilab/AnyDoor/blob/44ca2b2a70ec2cf107f3d26a5b46def6670fb0a5/LICENSE.txt): MIT, copyright DAMO Vision Intelligence Lab.
- The `dinov2/LICENSE` actually bundled in that pinned source declares **Attribution-NonCommercial4.0 International**. It is not replaced by the outer AnyDoor MIT notice. [Pinned bundled license](https://github.com/ali-vilab/AnyDoor/blob/44ca2b2a70ec2cf107f3d26a5b46def6670fb0a5/dinov2/LICENSE).
- Pretrained model weights may have separate terms. Retrieve and review them from their official sources. A code license alone does not establish weight/data/output redistribution rights.

## Data and qualitative examples

Real source photographs, synthetic images, per-image annotations/masks, full demo imagery and trained checkpoints are withheld in this public release. Dataset provenance and research access are not treated as automatic permission for public redistribution. Numerical experiment summaries and charts without underlying photographs are included.

Before adding images, pretrained/model files or a hosted inference service, separately confirm applicable rights, attribution and distribution obligations. This notice records observed component licenses and release scope; it is not a legal opinion or a license change for the existing project.
