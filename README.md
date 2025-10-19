# ThreatIntel-Andro: Expert-Verified Benchmarking for Robust Android Malware Research

## Overview
This repository accompanies the research paper *"hreatIntel-Andro: Expert-Verified Benchmarking for Robust Android Malware Research"*. We present **ThreatIntel-Andro**, an Android malware dataset grounded in real-world security industry expertise.

ThreatIntel-Andro is constructed by systematically collecting and analyzing professional threat reports published by leading cybersecurity vendors since 2016. From these reports, we extract **expert-validated indicators of compromise (IoCs)**—primarily file hashes and their associated malware family attributions. The final dataset comprises **5,123 malware samples** spanning **146 distinct malware families**.

All samples are retrieved from the [Koodous](https://koodous.com/) repository and temporally anchored using key metadata such as first submission dates, ensuring both timeliness and long-term research value.

### Long-Term Maintenance and Open Collaboration  
ThreatIntel-Andro is designed as a **long-term, openly maintained dataset** for the Android security research community. In future work, we plan to continuously expand the dataset with newly reported malware samples and refine family annotations based on evolving threat landscapes. We welcome **feedback and suggestions** from researchers in the field of Android malware detection to help improve the quality, coverage, and usability of the dataset.

### Request Access to the Dataset  
Due to legal and ethical considerations regarding malware distribution, full access to the ThreatIntel-Andro dataset is granted on a **research-purpose basis**.  

If you would like to access the dataset, please send an email to: **[bai931214@tju.edu.cn](mailto:bai931214@tju.edu.cn)**  

To ensure responsible usage, all requests will be **manually reviewed**. Please include the following information in your email:  
- The intended **purpose** of using the dataset  
- Your background in **Android security or malware research** (if applicable)  
- A link to your **recent publications**, **Google Scholar profile**, or personal research webpage  
- If you are a student, please include a link to your **advisor’s academic profile**  

We appreciate your cooperation and look forward to fostering collaborative, ethical research in the Android security community.

---

## Dataset and Resources

### 📁 `apks/`
- Contains a subset of the APK files we have downloaded.
- **Full APK access**: The complete set of APKs will be made publicly available for research purposes upon paper acceptance.

### 📁 `code_for_download_apks_from_koodous/`
- Includes custom scripts developed to:
  - Download APK files from Koodous.
  - Retrieve dynamic and static analysis reports from Koodous.

### 📁 `code_for_feature_extraction/`
- Contains code for extracting **static features** from APKs based on the **Drebin** feature extraction method.
- **Important update**: Recognizing that Android APIs and permissions have evolved significantly since the original Drebin study, we have updated the **permission-to-API mapping** to support API levels up to **level 36**.
- This enhancement ensures more **complete and accurate** feature extraction from modern malware samples, addressing limitations of older datasets that may miss contemporary malicious behaviors.
- The updated permission-to-API mappings are based on:  
  [https://github.com/XFY9326/AndroidInfo](https://github.com/XFY9326/AndroidInfo)

### 📁 `dynamic_analysis_feature/`
- Contains dynamic analysis reports for a subset of samples obtained from Koodous.
- **Full dynamic reports**: The complete set of dynamic analysis reports will be released upon paper acceptance.

### 📁 `experiments/`
- Contains code and scripts for reproducing the experiments presented in our paper.
- We attempted to benchmark several state-of-the-art Android malware classification methods, including **AndMFC** and **Meta-MAMC**.
- **Note on reproducibility**: The original source code for AndMFC and Meta-MAMC is either incomplete or not fully publicly available. In this work, we have **reimplemented and adapted** their approaches based on the published papers and available code fragments, making every effort to adhere to the original methodologies.
- Implementation details, hyperparameters, and evaluation scripts are included to ensure transparency and facilitate future comparisons.

### 📄 `ThreatIntel-Andro_report.csv`
- The core metadata file of the dataset.
- Contains the following fields for each sample:
  - `File hash` (SHA-256)
  - `Reported family`
  - `Source` 
  - `First Submission date` (to Virustotal)
  - `Report date` (publication date of the original analysis)
  - `Report URL` (link to the original vendor report)
  - `Category` (e.g., banking trojan, ransomware, spyware)
- Enables full **provenance tracking** and allows researchers to verify and audit each sample's origin and classification.

