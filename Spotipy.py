import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import random

auth_manager = SpotifyClientCredentials('','')
sp = spotipy.Spotify(auth_manager=auth_manager)

# Spotipy'nin erişim belirtecini alma ve gösterme
print(f"Access Token: {auth_manager.get_access_token()}")

# Spotify playlist'ten şarkı ID'lerini al
def getTrackIDs(playlist_id):
    track_ids = []
    playlist_items = sp.playlist_items(playlist_id, limit=50)
    for item in playlist_items['items']:
        track = item['track']
        if track:  # Track varsa
            track_ids.append(track['id'])
    return track_ids


def getTrackFeatures(track_id):
    track_info = sp.track(track_id)
    name = track_info.get('name', 'Unknown Name')
    album = track_info.get('album', {}).get('name', 'Unknown Album')
    artist = track_info.get('album', {}).get('artists', [{}])[0].get('name', 'Unknown Artist')
    url = track_info.get('external_urls', {}).get('spotify', 'No URL Available')
    return {"name": name, "album": album, "artist": artist, "url": url}


def get_emotion_recommendations(emotion_id):
    """
    Verilen emotion_id için Spotify'dan playlist önerileri ve şarkılar alır.
    """
    emotion_name = emotion_dict[emotion_id]
    print("Emotion is:", emotion_name)
    results = sp.search(q=emotion_name, type='playlist', limit=5)
    print(f"Spotify Search Results: {results}")
    
    if results['playlists']['items']:
        valid_playlists = [playlist for playlist in results['playlists']['items'] if playlist is not None]
        
        all_track_ids = []
        for playlist in valid_playlists:
            playlist_id = playlist['id']
            print(f"Playlist ID: {playlist_id}")
            
            track_ids = getTrackIDs(playlist_id)
            all_track_ids.extend(track_ids)  # Tüm şarkıları toplama
        
        if not all_track_ids:
            print("No tracks found in the playlists.")
            return []
    
        # Şarkıları shuffle yap
        random.shuffle(all_track_ids)
        
        # Şarkı özelliklerini al
        track_list = []
        for track_id in all_track_ids[:10]:  # İlk 10 şarkıyı al
            track_features = getTrackFeatures(track_id)
            print(f"Track Features: {track_features}")
            track_list.append({
                "name": track_features["name"],
                "album": track_features["album"],
                "artist": track_features["artist"],
                "url": track_features["url"]  # URL
            })
        return track_list
    
    print("No playlists found for the emotion.")
    return []



emotion_dict = {0:"Angry",1:"Disgusted",2:"Fearful",3:"Happy",4:"Neutral",5:"Sad",6:"Surprised"}

