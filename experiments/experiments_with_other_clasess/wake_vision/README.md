# Download Images for Building Wake Vision with Other Targets


## 🛠️ **Getting Started**

### Step 1: Install Docker Engine 🐋

First, install Docker on your machine:
- [Install Docker Engine](https://docs.docker.com/engine/install/).

---

### Step 2: Download the Target Images

Substitute **Gondola** with your target class (e.g. Dog, Cat, Bird...). The complete list of target classes can be found [here](https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions-boxable.csv).

```bash
sudo docker run -it --rm -v "$(pwd):/tmp" -w /tmp wake_vision:download python build_wake_vision_with_other_targets.py Gondola
```

Note: It does not apply Wake Vision pre-processing for standardized evaluation. See appendix J of Wake Vision paper for further information: https://arxiv.org/pdf/2405.00892

---
