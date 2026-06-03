import pyroomacoustics as pra
import numpy as np

def run_wargame_simulation(room_dim, participant_pos, mic_array_pos):
    # 1. Create the physical wargaming room (e.g., 10m x 8m x 3m)
    # rt60 sets the target reverberation decay time
    room = pra.ShoeBox(room_dim, fs=16000, max_order=12, rt60=0.5)

    # 2. Add a participant (e.g., Blue Team Commander)
    # We load a clean speech sample to test how it degrades in the space
    clean_speech, _ = pra.datasets.LinesClean().next()
    room.add_source(participant_pos, signal=clean_speech[0:16000])

    # 3. Add Microphone Array (e.g., 4 channels representing an MXA310)
    # Creating a small circular array layout to simulate directional capsules
    R = pra.circular_2d_array(center=mic_array_pos[:2], M=4, phi0=0, radius=0.05)
    # Add the Z height to the microphone array coordinates
    R = np.vstack((R, np.full((1, 4), mic_array_pos[2])))
    room.add_microphone_array(pra.MicrophoneArray(R, room.fs))

    # 4. Run Ray Tracing / Image Source Method
    room.simulate()
    
    # room.mic_array.signals contains the 4 channels of affected audio
    return room.mic_array.signals