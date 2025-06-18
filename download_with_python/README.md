**Download and Build Wake Vision Dataset with Python**

Follow the following steps to download and build the Wake Vision dataset with python. 

1. [Sign up](https://dataverse.harvard.edu/dataverseuser.xhtml;jsessionid=b78ff6ae13347e089bc776b916e9?editMode=CREATE&redirectPage=%2Fdataverse_homepage.xhtml) on Harvard Dataverse

2. On your account information page go to the API Token tab and create a new **API Token** for Harvard Dataverse

3. **Substitute "your-api-token-goes-here" with your API token in the following command** and run it inside the directory where you cloned this repository to download and build the Wake Vision Dataset:

```bash
python3 download_and_build_wake_vision_dataset.py your-api-token-goes-here
```

**Note**: Make sure to have at least 600 GB of free disk space.