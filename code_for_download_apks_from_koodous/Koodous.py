import os
import json
import logging
import pandas as pd
import requests
from time import sleep

# 1. Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# 2. Define custom exception for precise handling of API rate limit errors
class RateLimitException(Exception):
    """Custom exception for API rate limit errors"""
    pass

class KoodousAPI:
    """Wrapper for Koodous API to handle requests and responses"""
    def __init__(self, token):
        self.base_url = "https://developer.koodous.com"
        self.headers = {"Authorization": f"Token {token}"}

    def _handle_response(self, response):
        """Handle the HTTP response from the API"""
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 404:
            return None
        elif response.status_code == 204:
            # Analysis has started but not completed
            return "analysis_started"
        elif response.status_code == 429:
            # API rate limit error -> Raise custom exception
            error_detail = response.text
            raise RateLimitException(error_detail)
        else:
            logging.warning(f"⚠️ API request failed, status code {response.status_code}: {response.text}")
            return None

    def get_apk_info(self, sha256):
        """Get general information about the APK"""
        url = f"{self.base_url}/apks/{sha256}/"
        logging.info(f"🌐 Requesting APK information: {url}")
        response = requests.get(url, headers=self.headers)
        return self._handle_response(response)

    def get_apk_all_analysis(self, sha256):
        """Get all available analysis reports for the APK"""
        url = f"{self.base_url}/apks/{sha256}/analysis/"
        logging.info(f"📊 Requesting analysis reports: {url}")
        response = requests.get(url, headers=self.headers)
        return self._handle_response(response)
    
    def start_apk_static_analysis(self, sha256):
        """Submit a request to start static analysis of the APK"""
        url = f"{self.base_url}/apks/{sha256}/analyze_static/"
        logging.info(f"🔬 Submitting static analysis request: {url}")
        response = requests.post(url, headers=self.headers)
        return self._handle_response(response)

    def start_apk_dynamic_analysis(self, sha256):
        """Submit a request to start dynamic analysis of the APK"""
        url = f"{self.base_url}/apks/{sha256}/analyze_dynamic/"
        logging.info(f"🔍 Submitting dynamic analysis request: {url}")
        response = requests.post(url, headers=self.headers)
        return self._handle_response(response)
    
    def download_apk(self, sha256, save_dir):
        """Download the APK file"""
        url = f"{self.base_url}/apks/{sha256}/download/"
        logging.info(f"📥 Downloading APK: {url}")
        response = requests.get(url, headers=self.headers)
        
        # Handle download similarly, raise custom exception if rate limit is reached
        if response.status_code == 429:
            error_detail = response.text
            raise RateLimitException(f"🚫 APK download rate limit reached: {error_detail}")
        elif response.status_code != 200:
            raise Exception(f"Failed to download APK: {response.text}")
        
        file_path = os.path.join(save_dir, f"{sha256}.apk")
        with open(file_path, "wb") as file:
            file.write(response.content)
        return file_path

def mode_download_and_collect(api: KoodousAPI, api_counters, csv_file, apks_dir, reports_dir):
    """
    Mode 1: Download APKs and collect analysis results with intelligent merging strategy
    - Smartly checks Koodous for completed analysis to fill in null entries in the local JSON
    - Safely skips the current APK when rate limit errors occur, without saving changes
    """
    df = pd.read_csv(csv_file)

    os.makedirs(apks_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    for index, row in df.iterrows():
        sleep(0.3)  # Avoid making too many requests to the API too quickly
        sha256 = row["sha256"]
        logging.info(f"\n{'='*80}\n🔄 Processing APK [{index+1}/{len(df)}]: {sha256}\n{'='*80}")
        
        combined_file = os.path.join(reports_dir, f"{sha256}.json")
        apk_path = os.path.join(apks_dir, f"{sha256}.apk")

        try:
            # 1. If local cache exists, load it
            if os.path.exists(combined_file):
                logging.info(f"📂 Loading existing JSON data: {combined_file}")
                with open(combined_file, "r", encoding="utf-8") as f:
                    apk_data = json.load(f)
            else:
                logging.info(f"📂 No local data found, creating new record")
                apk_data = {"apk_info": None, "static_analysis": None, "dynamic_analysis": None}

            # 2. Check if any analysis part is missing locally
            is_static_missing_locally = apk_data.get("static_analysis") is None
            is_dynamic_missing_locally = apk_data.get("dynamic_analysis") is None

            if not is_static_missing_locally and not is_dynamic_missing_locally:
                logging.info("✅ All analysis data is complete in the local file")
            else:
                # 3. If local data is incomplete, query the API for the latest status
                logging.info("🔍 Local data is incomplete. Querying the API for the latest analysis status...")
                latest_info = api.get_apk_info(sha256)
                api_counters["apks_detail"] += 1

                if latest_info is None:
                    logging.warning(f"❌ APK not found on Koodous: {sha256}. Skipping.")
                    continue
                
                apk_data["apk_info"] = latest_info
                
                # 4. Decide whether to fetch complete analysis reports
                should_fetch_report = (is_static_missing_locally and latest_info.get("is_static_analyzed")) or \
                                      (is_dynamic_missing_locally and latest_info.get("is_dynamic_analyzed"))

                if should_fetch_report:
                    logging.info("📊 Analysis ready on Koodous server. Fetching full reports to fill missing parts...")
                    all_analysis = api.get_apk_all_analysis(sha256)
                    api_counters["analysis_reports"] += 1
                    
                    if all_analysis:
                        # 5. Smart merge: Only fill fields that are None
                        if is_static_missing_locally and ('cuckoo' in all_analysis or 'androguard' in all_analysis):
                            apk_data["static_analysis"] = {'cuckoo': all_analysis.get('cuckoo'), 'androguard': all_analysis.get('androguard')}
                            logging.info("    -> ✨ Successfully merged [Static Analysis]")
                        
                        if is_dynamic_missing_locally and 'droidbox' in all_analysis:
                            apk_data["dynamic_analysis"] = {'droidbox': all_analysis.get('droidbox')}
                            logging.info("    -> ✨ Successfully merged [Dynamic Analysis]")
                else:
                    logging.info("⏳ Analysis is not yet complete on Koodous server or exists locally. Skipping fetch.")
            
            # Save the updated data if no errors occurred
            with open(combined_file, "w", encoding="utf-8") as f:
                json.dump(apk_data, f, indent=4, ensure_ascii=False)
            logging.info(f"💾 Successfully saved updated data to: {combined_file}")

            # Lastly, check and download APK if missing
            if not os.path.exists(apk_path):
                logging.info(f"📥 APK not found locally, starting download...")
                api.download_apk(sha256, apks_dir)
                api_counters["apks_downloads"] += 1
                logging.info(f"✅ Successfully downloaded APK (Total downloads: {api_counters['apks_downloads']})")
            else:
                logging.info(f"✅ APK already exists locally, skipping download")

        except RateLimitException as e:
            # Specially catch rate limit errors
            logging.error(f"🚫 API rate limit reached: {sha256}. Details: {e}")
            logging.warning(f"🛑 Stopping processing for this APK. JSON file will not be modified.")
            continue  # Proceed to the next APK
            
        except Exception as e:
            # Catch any other unexpected errors
            logging.error(f"💥 Unexpected error while processing {sha256}: {e}")

def mode_submit_analysis(api: KoodousAPI, api_counters, csv_file, reports_dir):
    """
    Mode 2: Check and submit missing analysis requests
    - Does not download APK files
    - Retrieves APK information to check analysis status
    - Submits analysis requests for APKs without results
    """
    df = pd.read_csv(csv_file)
    os.makedirs(reports_dir, exist_ok=True)

    for index, row in df.iterrows():
        sha256 = row["sha256"]
        logging.info(f"\n{'='*60}\n🧪 Analysis Mode [{index+1}/{len(df)}]: {sha256}\n{'='*60}")
        
        try:
            info = api.get_apk_info(sha256)
            api_counters["apks_detail"] += 1

            if info is None:
                logging.warning(f"❌ APK not found on Koodous: {sha256}")
                continue

            has_static = info.get("is_static_analyzed", False)
            has_dynamic = info.get("is_dynamic_analyzed", False)
            logging.info(f"📊 Analysis status - Static: {'✅' if has_static else '❌'}, Dynamic: {'✅' if has_dynamic else '❌'}")

            if not has_static:
                logging.info(f"🔬 Submitting static analysis request...")
                result = api.start_apk_static_analysis(sha256)
                api_counters["analysis_requests"] += 1
                logging.info(f"➡️ Static analysis response: {result}")

            if not has_dynamic:
                logging.info(f"🔍 Submitting dynamic analysis request...")
                result = api.start_apk_dynamic_analysis(sha256)
                api_counters["analysis_requests"] += 1
                logging.info(f"➡️ Dynamic analysis response: {result}")

        except RateLimitException as e:
            logging.error(f"🚫 API rate limit reached: {sha256}. Details: {e}")
            logging.warning(f"🛑 Stopping processing for this APK.")
            continue
        except Exception as e:
            logging.error(f"💥 Error while processing {sha256}: {e}")


def main():
    """
    Koodous API Dual Mode Processor
    - Mode 1 (download_and_collect): Download APKs and collect analysis results
    - Mode 2 (submit_analysis): Check and submit missing analysis requests
    """
    # --- Configuration ---
    token = "XXXX"  # <-- Important: Replace with your API token
    csv_file = "XXXXX"  # <-- Set your CSV file path
    apks_dir = "XXXXX" # <-- Set your APK download directory
    reports_dir = "XXXXX"  # <-- Set your analysis reports directory 
    run_mode = "download_and_collect"  # Options: "download_and_collect" or "submit_analysis"
    # --- End of Configuration ---

    api = KoodousAPI(token)
    api_counters = {
        "apks_detail": 0,
        "apks_downloads": 0,
        "analysis_reports": 0,
        "analysis_requests": 0
    }

    logging.info(f"🚀 Koodous Processor started in '{run_mode}' mode")

    if run_mode == "download_and_collect":
        mode_download_and_collect(api, api_counters, csv_file, apks_dir, reports_dir)
    elif run_mode == "submit_analysis":
        mode_submit_analysis(api, api_counters, csv_file, reports_dir)
    else:
        logging.error(f"❌ Unknown mode: {run_mode}")
        return

    # Display API usage statistics
    logging.info(f"\n{'='*100}")
    logging.info(f"🎉 Processing completed in '{run_mode}' mode. API usage summary:")
    logging.info(f"{'='*100}")
    logging.info(f"📊 APK details: {api_counters['apks_detail']}")
    logging.info(f"📥 APK downloads: {api_counters['apks_downloads']}")
    logging.info(f"📋 Analysis reports: {api_counters['analysis_reports']}")
    logging.info(f"🔬 Analysis requests: {api_counters['analysis_requests']}")
    logging.info(f"🔢 Total API calls: {sum(api_counters.values())}")
    logging.info(f"{'='*100}")

if __name__ == "__main__":
    main()
