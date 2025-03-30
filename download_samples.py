import os
import requests
import logging
from tqdm import tqdm
import urllib.parse
import zipfile
import tarfile
import shutil
import time
import argparse

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Updated source for samples
DATASET_SOURCES = [
    {
        "name": "FMA Small",
        "url": "https://os.unil.cloud.switch.ch/fma/fma_small.zip",
        "extract_path": "datasets/fma_small",
        "genres_mapping": {
            "Electronic": "electronic",
            "Experimental": "electronic",
            "Folk": "folk",
            "Hip-Hop": "hiphop",
            "Instrumental": "instrumental",
            "International": "world",
            "Pop": "pop",
            "Rock": "rock",
            "Jazz": "jazz",
            "Blues": "blues",
            "Classical": "classical",
            "Country": "country",
            "Disco": "disco",
            "Metal": "metal",
            "Reggae": "reggae"
        },
    },
    {
        "name": "GTZAN (alternate source)",
        "url": "https://zenodo.org/record/4594877/files/GTZAN-all-audio.zip",
        "extract_path": "datasets/gtzan",
        "genres_mapping": {
            "blues": "blues",
            "classical": "classical",
            "country": "country",
            "disco": "disco",
            "hiphop": "hiphop",
            "jazz": "jazz",
            "metal": "metal",
            "pop": "pop",
            "reggae": "reggae",
            "rock": "rock"
        },
    }
]

# Define some alternate free sources for individual genres
FALLBACK_GENRE_SAMPLES = {
    "classical": [
        "https://archive.org/download/ClassicalMusicMP3/Bach-AirOnTheGString.mp3",
        "https://archive.org/download/ClassicalMusicMP3/Beethoven-FurElise.mp3",
        "https://archive.org/download/ClassicalMusicMP3/Beethoven-Moonlight_Sonata.mp3",
        "https://archive.org/download/ClassicalMusicMP3/Mozart-Lacrimosa.mp3",
        "https://archive.org/download/ClassicalMusicMP3/Mozart-RequiemInD.mp3",
        "https://archive.org/download/3-beethoven-piano-sonatas-pathetique-moonlight-appassionata/01_beethoven_pathetique_1st_movement.mp3",
        "https://archive.org/download/free-great-piano-works-of-mozart-chopin-debussy/Chopin-Nocturne-Op9-No2.mp3",
        "https://archive.org/download/cd_masterworks-of-the-classical-era_academy-of-london-haydn-handel-mozart/disc1/03.%20Academy%20Of%20London%20-%20Introduction%3A%20Grave%3B%20Allegro%20From%20the%20Overture%20to%20the%20Occasional%20Oratorio%2C%20HWV.62_sample.mp3"
    ],
    "jazz": [
        "https://archive.org/download/78_west-end-blues_louis-armstrong-and-his-hot-five-louis-armstrong-fred-robinson-jimmy_gbia0071396b/01%20-%20West%20End%20Blues%20-%20Louis%20Armstrong%20and%20his%20Hot%20Five-restored.mp3",
        "https://archive.org/download/78_body-and-soul_coleman-hawkins-and-his-orchestra-johnny-green-edward-heyman-robert-sa_gbia0066002a/Body%20and%20Soul%20-%20Coleman%20Hawkins%20and%20his%20Orchestra-restored.mp3",
        "https://archive.org/download/78_st-louis-blues_bessie-smith-bessie-smith-and-her-band-louis-armstrong-fred-longshaw_gbia0075287a/St.%20Louis%20Blues%20-%20Bessie%20Smith%20-%20Bessie%20Smith%20and%20her%20Band-restored.mp3",
        "https://archive.org/download/78_caravan_duke-ellington-and-his-famous-orchestra-duke-ellington-mills-juan-tizol_gbia0030336b/Caravan%20-%20Duke%20Ellington%20and%20his%20Famous%20Orchestra-restored.mp3",
        "https://archive.org/download/78_sophisticated-lady_duke-ellington-and-his-famous-orchestra-duke-ellington-irving-m_gbia0031101b/Sophisticated%20Lady%20-%20Duke%20Ellington%20and%20his%20Famous%20Orchestra-restored.mp3",
        "https://archive.org/download/cd_miles-davis-birdland-1951_miles-davis-quintet/disc1/01.%20Miles%20Davis%20Quintet%20-%20Morpheus_sample.mp3",
        "https://archive.org/download/cd_miles-davis-birdland-1951_miles-davis-quintet/disc1/04.%20Miles%20Davis%20Quintet%20-%20Ray's%20Idea_sample.mp3"
    ],
    "rock": [
        "https://archive.org/download/cd_woodstock-three-days-of-peace-and-music-the_various-artists/disc1/11.%20Various%20Artists%20-%20I%20Had%20a%20Dream_sample.mp3",
        "https://archive.org/download/cd_woodstock-three-days-of-peace-and-music-the_various-artists/disc1/08.%20Various%20Artists%20-%20Soul%20Sacrifice_sample.mp3",
        "https://archive.org/download/cd_woodstock-three-days-of-peace-and-music-the_various-artists/disc1/04.%20Various%20Artists%20-%20Coming%20into%20Los%20Angeles_sample.mp3",
        "https://archive.org/download/cd_woodstock-three-days-of-peace-and-music-the_various-artists/disc2/01.%20Various%20Artists%20-%20Going%20Up%20the%20Country_sample.mp3",
        "https://archive.org/download/cd_grateful-dead_grateful-dead/disc1/02.%20Grateful%20Dead%20-%20Cold%20Rain%20And%20Snow_sample.mp3",
        "https://archive.org/download/cd_grateful-dead_grateful-dead/disc1/08.%20Grateful%20Dead%20-%20Cream%20Puff%20War_sample.mp3",
        "https://archive.org/download/TheVelvetUndergroundTheCompleteMatrixTapes/Disc%201/01%20-%20I'm%20Waiting%20For%20The%20Man%20%28Second%20Show%29.mp3",
        "https://archive.org/download/cd_this-is-psychedelic-rock_purifying-elements-the-crystal-set-die-warzau-/disk1/01.%20Various%20Artists%20-%20Dreaming_sample.mp3"
    ],
    "pop": [
        "https://archive.org/download/cd_pop-memories-of-the-60s-bring-back_various-artists-bobby-vee-lesley-gore-del/disc1/03.%20Various%20Artists%20-%20The%20Night%20Has%20a%20Thousand%20Eyes_sample.mp3",
        "https://archive.org/download/cd_pop-memories-of-the-60s-bring-back_various-artists-bobby-vee-lesley-gore-del/disc1/01.%20Various%20Artists%20-%20The%20Night%20Has%20a%20Thousand%20Eyes_sample.mp3",
        "https://archive.org/download/78_alexander-ragtime-band-medley_bessie-smith-bessie-smith-and-her-band-fred-longsha_gbia0066553a/Alexander%20Ragtime%20Band%20Medley%20-%20Bessie%20Smith%20-%20Bessie%20Smith%20and%20her%20Band-restored.mp3",
        "https://archive.org/download/cd_the-pye-singles-collection_doris-day/disc1/09.%20Doris%20Day%20-%20Everybody%20Loves%20a%20Lover_sample.mp3",
        "https://archive.org/download/cd_greatest-hits_duran-duran/disc1/05.%20Duran%20Duran%20-%20New%20Moon%20on%20Monday_sample.mp3",
        "https://archive.org/download/cd_the-greatest-hits-of-the-80s_abc-adam-ant-animotion-after-the-fire-bananarama/disc1/01.%20Various%20Artists%20-%20The%20Look%20of%20Love%20%28Part%20One%29_sample.mp3"
    ],
    "blues": [
        "https://archive.org/download/78_stormy-monday-blues_jay-mcshann-and-his-orchestra-j-witherspoon-e-humes_gbia0105975a/01%20-%20Stormy%20Monday%20Blues%20-%20Jay%20McShann%20and%20his%20Orchestra-restored.mp3",
        "https://archive.org/download/78_crossroad-blues_robert-johnson-robert-johnson_gbia0023863a/Crossroad%20Blues%20-%20Robert%20Johnson-restored.mp3",
        "https://archive.org/download/78_memphis-blues_louis-armstrong-and-his-orchestra-louis-armstrong-w-c-handy_gbia0012954b/02%20-%20Memphis%20Blues%20-%20Louis%20Armstrong%20and%20his%20Orchestra-restored.mp3",
        "https://archive.org/download/cd_lead-belly-king-of-the-twelve-string-guitar_lead-belly/disc1/06.%20Lead%20Belly%20-%20New%20Orleans%20%28Birmingham%20Blues%29_sample.mp3",
        "https://archive.org/download/cd_mississippi-blues_mississippi-john-hurt/disc1/01.%20Mississippi%20John%20Hurt%20-%20Got%20the%20Blues%20%28Can%27t%20Be%20Satisfied%29_sample.mp3",
        "https://archive.org/download/78_dust-my-broom_elmore-james-and-his-broomdusters_gbia0242139a/01%20-%20Dust%20My%20Broom%20-%20Elmore%20James%20and%20his%20Broomdusters-restored.mp3"
    ],
    "country": [
        "https://archive.org/download/cd_honky-tonk-heroes_willie-nelson-waylon-jennings-johnny-cash-kris-/disc1/06.%20Various%20Artists%20-%20For%20the%20Good%20Times_sample.mp3",
        "https://archive.org/download/cd_honky-tonk-heroes_willie-nelson-waylon-jennings-johnny-cash-kris-/disc1/08.%20Various%20Artists%20-%20Me%20and%20Paul_sample.mp3",
        "https://archive.org/download/cd_im-gonna-hire-a-wino-to-decorate-our-home-and_david-frizzell/disc1/04.%20David%20Frizzell%20-%20I%27m%20Gonna%20Hire%20A%20Wino%20To%20Decorate%20Our%20Home_sample.mp3",
        "https://archive.org/download/cd_the-essential-johnny-cash_johnny-cash/disc1/08.%20Johnny%20Cash%20-%20I%20Walk%20the%20Line_sample.mp3",
        "https://archive.org/download/cd_the-essential-johnny-cash_johnny-cash/disc1/12.%20Johnny%20Cash%20-%20Ring%20of%20Fire_sample.mp3"
    ],
    "electronic": [
        "https://archive.org/download/cd_freezone-twelve-miles-above-the-surface_deep-space-network-dave-clarke-gerd-coldcu/disc1/01.%20Various%20Artists%20-%20Aqua%20Sphere_sample.mp3",
        "https://archive.org/download/cd_freezone-twelve-miles-above-the-surface_deep-space-network-dave-clarke-gerd-coldcu/disc1/03.%20Various%20Artists%20-%20Nightime_sample.mp3",
        "https://archive.org/download/cd_orbital-20_orbital/disc1/01.%20Orbital%20-%20Lush%203_sample.mp3",
        "https://archive.org/download/cd_abstract-dimension_various-artists/disc1/03.%20Various%20Artists%20-%20Time%20For%20Nothing_sample.mp3",
        "https://archive.org/download/cd_abstract-dimension_various-artists/disc1/04.%20Various%20Artists%20-%20Future%20Trance_sample.mp3",
        "https://archive.org/download/cd_synthesis-resonance_the-cyborg_team_-_cybernetic_djs-_/disc1/01.%20Various%20Artists%20-%20Night%20Cruiser_sample.mp3"
    ],
    "hiphop": [
        "https://archive.org/download/cd_just-a-poet-with-soul_gil-scott-heron/disc1/01.%20Gil%20Scott-Heron%20-%20The%20Revolution%20Will%20Not%20Be%20Televised_sample.mp3",
        "https://archive.org/download/cd_just-a-poet-with-soul_gil-scott-heron/disc1/03.%20Gil%20Scott-Heron%20-%20Home%20Is%20Where%20The%20Hatred%20Is_sample.mp3",
        "https://archive.org/download/cd_hip-hop-and-the-world-we-live-in_various/disc1/01.%20Various%20Artists%20-%20Down%20in%20the%20Dirty_sample.mp3",
        "https://archive.org/download/cd_hop-hop-and-the-world-we-live-in_various-artists/disc1/01.%20Various%20Artists%20-%20Down%20in%20the%20Dirty_sample.mp3",
        "https://archive.org/download/cd_hip-hop-the-collection_various-artists/disc1/05.%20Various%20Artists%20-%20Slam_sample.mp3"
    ],
    "metal": [
        "https://archive.org/download/cd_master-of-reality_black-sabbath/disc1/02.%20Black%20Sabbath%20-%20After%20Forever_sample.mp3",
        "https://archive.org/download/cd_master-of-reality_black-sabbath/disc1/06.%20Black%20Sabbath%20-%20Lord%20of%20this%20World_sample.mp3",
        "https://archive.org/download/cd_master-of-reality_black-sabbath/disc1/07.%20Black%20Sabbath%20-%20Solitude_sample.mp3",
        "https://archive.org/download/cd_system-of-a-down_system-of-a-down/disc1/05.%20System%20of%20a%20Down%20-%20Pluck_sample.mp3",
        "https://archive.org/download/cd_system-of-a-down_system-of-a-down/disc1/08.%20System%20of%20a%20Down%20-%20War%3F_sample.mp3"
    ],
    "reggae": [
        "https://archive.org/download/cd_roots-of-reggae_peter-tosh-jimmy-cliff-lee-perry-desmond-dekker/disc1/01.%20Various%20Artists%20-%20400%20Years_sample.mp3",
        "https://archive.org/download/cd_roots-of-reggae_peter-tosh-jimmy-cliff-lee-perry-desmond-dekker/disc1/03.%20Various%20Artists%20-%20Equal%20Rights_sample.mp3",
        "https://archive.org/download/cd_roots-of-reggae_peter-tosh-jimmy-cliff-lee-perry-desmond-dekker/disc1/10.%20Various%20Artists%20-%20No%20Sympathy_sample.mp3",
        "https://archive.org/download/cd_roots-of-reggae_peter-tosh-jimmy-cliff-lee-perry-desmond-dekker/disc1/12.%20Various%20Artists%20-%20Mystery%20Babylon_sample.mp3",
        "https://archive.org/download/cd_roots-of-reggae_peter-tosh-jimmy-cliff-lee-perry-desmond-dekker/disc1/18.%20Various%20Artists%20-%20Waterfall_sample.mp3"
    ],
    "folk": [
        "https://archive.org/download/cd_this-land-is-your-land-the-asch-recordings-v_woody-guthrie/disc1/01.%20Woody%20Guthrie%20-%20This%20Land%20Is%20Your%20Land_sample.mp3",
        "https://archive.org/download/cd_this-land-is-your-land-the-asch-recordings-v_woody-guthrie/disc1/08.%20Woody%20Guthrie%20-%20Talking%20Fishing%20Blues_sample.mp3",
        "https://archive.org/download/cd_time-the-revelator_gillian-welch/disc1/01.%20Gillian%20Welch%20-%20Revelator_sample.mp3",
        "https://archive.org/download/cd_time-the-revelator_gillian-welch/disc1/03.%20Gillian%20Welch%20-%20Red%20Clay%20Halo_sample.mp3",
        "https://archive.org/download/cd_time-the-revelator_gillian-welch/disc1/08.%20Gillian%20Welch%20-%20Dear%20Someone_sample.mp3"
    ],
    "world": [
        "https://archive.org/download/cd_authentic-middle-east_various-artists/disc1/05.%20Various%20Artists%20-%20Azeri%20Dance%20%28Azerbaijan%29_sample.mp3",
        "https://archive.org/download/cd_authentic-middle-east_various-artists/disc1/08.%20Various%20Artists%20-%20The%20Snake%20Dance%20%28Middle%20East%29_sample.mp3",
        "https://archive.org/download/cd_authentic-middle-east_various-artists/disc1/14.%20Various%20Artists%20-%20Andalusian%20Dance%20%28Morocco%29_sample.mp3",
        "https://archive.org/download/cd_a-music-tapestry-of-africa_various-artists/disc1/18.%20Various%20Artists%20-%20Sikiza%20Ndebe%20Yangu_sample.mp3",
        "https://archive.org/download/cd_cuba_francisco-aguabella-afro-cuban-ritual-dru/disc1/04.%20Francisco%20Aguabella%20Afro-Cuban%20Ritual%20Drummers%20-%20Obatala_sample.mp3"
    ]
}

def download_with_progress(url, dest_path, chunk_size=1024*1024):
    """Download a file with progress bar"""
    logger.info(f"Downloading from {url}")
    
    r = requests.get(url, stream=True)
    total_size = int(r.headers.get('content-length', 0))
    
    with open(dest_path, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=os.path.basename(dest_path)) as pbar:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    
    return dest_path

def download_file(url, save_path, max_retries=3):
    """Download a file from a URL to a local path with retries"""
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Try to download the file with retries
        for attempt in range(max_retries):
            try:
                download_with_progress(url, save_path)
                return True
            except Exception as e:
                logger.warning(f"Download attempt {attempt+1} failed: {str(e)}")
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    raise
        
    except Exception as e:
        logger.error(f"Error downloading {url}: {str(e)}")
        return False

def extract_archive(archive_path, extract_path):
    """Extract a zip or tar.gz archive to the specified path"""
    os.makedirs(extract_path, exist_ok=True)
    
    try:
        if archive_path.endswith('.zip'):
            logger.info(f"Extracting zip file: {archive_path}")
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(extract_path)
        elif archive_path.endswith('.tar.gz') or archive_path.endswith('.tgz'):
            logger.info(f"Extracting tar.gz file: {archive_path}")
            with tarfile.open(archive_path, 'r:gz') as tar_ref:
                tar_ref.extractall(extract_path)
        else:
            logger.error(f"Unsupported archive format: {archive_path}")
            return False
            
        logger.info(f"Extraction complete: {archive_path} -> {extract_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error extracting {archive_path}: {str(e)}")
        return False

def organize_samples_by_genre(extract_path, genres_mapping, dataset_name):
    """Organize extracted files into genre folders"""
    samples_dir = "samples"
    os.makedirs(samples_dir, exist_ok=True)
    
    # Different datasets have different organization, handle each specifically
    if dataset_name == "FMA Small":
        # Process FMA dataset
        logger.info("Organizing FMA dataset...")
        tracks_file = os.path.join(extract_path, "fma_metadata", "tracks.csv")
        
        # Map file IDs to genres
        if os.path.exists(tracks_file):
            import pandas as pd
            
            try:
                # Load metadata
                tracks = pd.read_csv(tracks_file, index_col=0, header=[0, 1])
                
                # Get genre information
                genre_col = tracks.loc[:, ('track', 'genre_top')]
                
                # Process each audio file
                audio_dir = os.path.join(extract_path, "fma_small")
                file_count = 0
                
                for root, dirs, files in os.walk(audio_dir):
                    for file in files:
                        if file.endswith('.mp3'):
                            try:
                                # Extract track ID from filename (e.g., 000001.mp3 -> 1)
                                track_id = int(os.path.splitext(file)[0])
                                
                                # Get genre for this track
                                genre = genre_col.get(track_id)
                                
                                if genre in genres_mapping:
                                    target_genre = genres_mapping[genre]
                                    
                                    # Create genre directory if needed
                                    genre_dir = os.path.join(samples_dir, target_genre)
                                    os.makedirs(genre_dir, exist_ok=True)
                                    
                                    # Copy file to genre directory
                                    source_path = os.path.join(root, file)
                                    target_path = os.path.join(genre_dir, file)
                                    
                                    if not os.path.exists(target_path):
                                        shutil.copy2(source_path, target_path)
                                        file_count += 1
                            except:
                                continue
                
                logger.info(f"Organized {file_count} files from FMA dataset")
                
            except Exception as e:
                logger.error(f"Error processing FMA metadata: {str(e)}")
        
    elif dataset_name == "GTZAN (alternate source)":
        # Process GTZAN dataset
        logger.info("Organizing GTZAN dataset...")
        file_count = 0
        
        # GTZAN has folders for each genre
        for genre in os.listdir(extract_path):
            genre_path = os.path.join(extract_path, genre)
            
            if os.path.isdir(genre_path) and genre.lower() in genres_mapping:
                target_genre = genres_mapping[genre.lower()]
                
                # Create genre directory if needed
                genre_dir = os.path.join(samples_dir, target_genre)
                os.makedirs(genre_dir, exist_ok=True)
                
                # Copy all mp3/wav files
                for file in os.listdir(genre_path):
                    if file.endswith(('.mp3', '.wav', '.au')):
                        source_path = os.path.join(genre_path, file)
                        target_path = os.path.join(genre_dir, f"gtzan_{file}")
                        
                        if not os.path.exists(target_path):
                            shutil.copy2(source_path, target_path)
                            file_count += 1
        
        logger.info(f"Organized {file_count} files from GTZAN dataset")
    
    else:
        logger.warning(f"Unknown dataset format: {dataset_name}")

def download_fallback_samples():
    """Download individual samples as fallback option"""
    samples_dir = "samples"
    os.makedirs(samples_dir, exist_ok=True)
    
    successful_downloads = 0
    total_files = sum(len(urls) for urls in FALLBACK_GENRE_SAMPLES.values())
    
    logger.info(f"Downloading {total_files} individual sample files...")
    
    for genre, urls in FALLBACK_GENRE_SAMPLES.items():
        # Create genre directory
        genre_dir = os.path.join(samples_dir, genre)
        os.makedirs(genre_dir, exist_ok=True)
        
        # Download each sample
        for url in urls:
            # Extract filename from URL or generate a unique one
            filename = os.path.basename(urllib.parse.unquote(url))
            if '?' in filename:  # Remove URL parameters if present
                filename = filename.split('?')[0]
                
            save_path = os.path.join(genre_dir, filename)
            
            if os.path.exists(save_path):
                logger.info(f"File already exists: {save_path}")
                successful_downloads += 1
                continue
            
            logger.info(f"Downloading {url}...")
            success = download_file(url, save_path)
            
            if success:
                successful_downloads += 1
    
    logger.info(f"Downloaded {successful_downloads}/{total_files} individual sample files.")
    return successful_downloads

def has_sufficient_samples():
    """Check if we have sufficient samples for each genre"""
    samples_dir = "samples"
    
    if not os.path.exists(samples_dir):
        return False, []
    
    # Check for each major genre
    min_samples = 10
    genres_with_samples = []
    insufficient_genres = []
    
    for genre in MAJOR_GENRES:
        genre_dir = os.path.join(samples_dir, genre)
        if not os.path.exists(genre_dir):
            insufficient_genres.append((genre, 0))
            continue
            
        audio_files = [f for f in os.listdir(genre_dir) 
                      if f.endswith(('.mp3', '.wav', '.ogg'))
                      and not f.startswith('.')]
        
        if len(audio_files) >= min_samples:
            genres_with_samples.append(genre)
        else:
            insufficient_genres.append((genre, len(audio_files)))
    
    if insufficient_genres:
        genres_info = ", ".join([f"{g} ({n} samples)" for g, n in insufficient_genres])
        logger.info(f"Insufficient samples for genres: {genres_info}")
        return False, insufficient_genres
    
    return True, []

def download_samples(force=False):
    """Download music samples for genre classification"""
    logger.info("Starting download of music datasets")
    
    # Force is now a parameter so we can force download from command line
    if not force:
        # Check if we already have sufficient samples
        has_samples, _ = has_sufficient_samples()
        if has_samples:
            logger.info("Sufficient samples already exist. Skipping downloads.")
            return True
    
    # Create samples directory if it doesn't exist
    samples_dir = "samples"
    os.makedirs(samples_dir, exist_ok=True)
    
    # Try downloading individual samples as fallback
    logger.info("Downloading individual genre samples from fallback sources")
    success = download_fallback_samples()
    
    # Check if we now have sufficient samples
    has_samples, _ = has_sufficient_samples()
    
    return has_samples or success

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download music samples for genre classification')
    parser.add_argument('--force', action='store_true', help='Force download even if samples exist')
    args = parser.parse_args()
    
    download_samples(force=args.force) 