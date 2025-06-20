
# Build Visual Wake Words with Other Targets


## 🛠️ **Getting Started**

### Step 1: Install Docker Engine 🐋

First, install Docker on your machine:
- [Install Docker Engine](https://docs.docker.com/engine/install/).

---

### Step 2: Download the Target Images

Substitute **bird** with your target class (e.g. dog, cat, ship...). The complete list of target classes can be found [here](https://cocodataset.org/#explore).

```bash
sudo docker run -it --rm -v "$(pwd):/tmp" -w /tmp wake_vision:download python build_vww_with_arbitrary_class.py bird
```
---