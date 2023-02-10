import base64
import pandas as pd


def create_onedrive_directdownload(onedrive_link):
    data_bytes64 = base64.b64encode(bytes(onedrive_link, 'utf-8'))
    data_bytes64_String = data_bytes64.decode('utf-8').replace('/', '_').replace('+', '-').rstrip("=")
    resultUrl = f"https://api.onedrive.com/v1.0/shares/u!{data_bytes64_String}/root/content"
    return resultUrl


# Input any OneDrive URL
onedrive_url = "https://1drv.ms/x/s!AmLiprCs46qqhMgFb5pcJisePNNlXw?e=ZgPsGZ"
onedrive_url = "https://onedrive.live.com/?authkey=%21ALdRo7h4KUAJoxc&id=B232B60CFF808B67%2117694&cid=B232B60CFF808B67"
# Generate Direct Download URL from above Script
direct_download_url = create_onedrive_directdownload(onedrive_url)
print(direct_download_url)
# Load Dataset to the Dataframe
df = pd.read_excel(direct_download_url)
print(df.head())
