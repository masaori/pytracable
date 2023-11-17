from dataclasses import dataclass, asdict
from datetime import datetime
import os
from typing import Optional, List, Any
import json
from abc import ABC, abstractmethod
import subprocess
import traceback
# pip
from ytmusicapi import YTMusic  # type: ignore
import essentia.standard as es  # type: ignore
import librosa
import numpy as np
import soundfile as sf  # type: ignore
import pychorus  # type: ignore
import demucs.separate


#
# Base Entity
#


@dataclass
class EntityBase:
    def to_dict(self):
        return asdict(self)

    def __str__(self):
        return json.dumps(self.to_dict())


@dataclass
class Datatype:
    name: str
    sub_type_name: Optional[str] = None

    def __init__(self, name: str, sub_type_name: Optional[str] = None):
        self.name = name
        self.sub_type_name = sub_type_name

    def __eq__(self, __value: object) -> bool:
        if isinstance(__value, Datatype):
            if self.sub_type_name is None and __value.sub_type_name is None:
                return self.name == __value.name
            else:
                return self.name == __value.name and self.sub_type_name == __value.sub_type_name
        else:
            return False

    def __str__(self):
        if self.sub_type_name is None:
            return f"{self.name}"
        else:
            return f"{self.name}<{self.sub_type_name}>"

    def __hash__(self):
        if self.sub_type_name is None:
            return hash(self.name)
        else:
            return hash((self.name, self.sub_type_name))


#
# Entity
#


@dataclass
class ProcessCategoryDefinition(EntityBase):
    id: str


@dataclass
class ProcessDefinition(EntityBase):
    id: str
    process_category_id: str
    input_data_types: List[Datatype]
    output_data_types: List[Datatype]
    append_id_as_suffix: bool = False

    def suffix(self, id: str) -> str:
        return f"{id}_{self.id}" if self.append_id_as_suffix else f"{self.process_category_id}_{self.id}"


@dataclass
class TransitionDefinition(EntityBase):
    source_process_id: str
    target_process_id: str
    prerequisite_process_ids: List[str]

    def __str__(self):
        return f"({self.source_process_id}->{self.target_process_id})"


@dataclass
class WorkflowDefinition(EntityBase):
    id: str
    start_process_id: str


# @dataclass
# class WorkflowConfiguration(EntityBase):
#     id: str
#     workflow_id: str
#     force_all: bool


# @dataclass
# class WorkflowProcessConfiguration(EntityBase):
#     id: str
#     workflow_id: str
#     process_id: str
#     force: bool


def check_no_cycles(start_process_id, transitions) -> bool:
    def dfs(current_process_id, visited):
        if current_process_id in visited:
            return False

        visited.add(current_process_id)
        next_processes = [
            t.target_process_id for t in transitions if t.source_process_id == current_process_id]
        for next_id in next_processes:
            if not dfs(next_id, visited):
                return False

        visited.remove(current_process_id)
        return True

    return dfs(start_process_id, set())


def check_transition_data_types(transitions, processes) -> Optional[str]:
    for transition in transitions:
        try:
            source_process = processes[transition.source_process_id]
        except KeyError:
            return f"Transition {transition} is invalid. Source process {transition.source_process_id} is not found."

        try:
            target_process = processes[transition.target_process_id]
        except KeyError:
            return f"Transition {transition} is invalid. Target process {transition.target_process_id} is not found."

        if set(source_process.output_data_types) != set(target_process.input_data_types):
            return f"Transition {transition} is invalid. {source_process.output_data_types} -> {target_process.input_data_types}"

    return None


def check_workflow_validity(start_process_id, processes, transitions) -> Optional[str]:
    if not check_no_cycles(start_process_id, transitions):
        return "Workflow has cycles."

    return check_transition_data_types(transitions, processes)


def generate_mermaid_diagram(
    workflow_definition: WorkflowDefinition,
    process_category_definitions: List[ProcessCategoryDefinition],
    process_definitions: List[ProcessDefinition],
    transition_definitions: List[TransitionDefinition]
) -> str:
    mermaid_str = "graph LR\n"

    # カテゴリごとにサブグラフを作成
    for category in process_category_definitions:
        mermaid_str += f"    subgraph {category.id}\n"
        for process in [p for p in process_definitions if p.process_category_id == category.id]:
            node_label = f"{process.id}({', '.join(dt.__str__() for dt in process.input_data_types)} -> {', '.join(dt.__str__() for dt in process.output_data_types)})"
            mermaid_str += f"        {process.id}[\"{node_label}\"]\n"
        mermaid_str += "    end\n"

    # トランジションを追加
    for transition in transition_definitions:
        mermaid_str += f"    {transition.source_process_id} --> {transition.target_process_id}\n"

    return mermaid_str


@dataclass
class FileDetail:
    location: str
    key: str

    def to_dict(self):
        return asdict(self)


@dataclass
class SongInfo:
    title: str
    artist: str

    def to_dict(self):
        return asdict(self)


@dataclass
class ProcessRecord:
    process_id: str
    started_at: str  # ISO8601
    finished_at: str  # ISO8601
    elapsed_secs: float  # seconds
    error: Optional[str] = None

    def to_dict(self):
        return asdict(self)


@dataclass
class MetaInfo:
    id: str
    song_info: Optional[SongInfo]
    file_details: List[FileDetail]
    process_records: List[ProcessRecord]
    discarded_reason: Optional[str] = None

    @staticmethod
    def default(id: str):
        return MetaInfo(
            id=id,
            song_info=None,
            file_details=[],
            process_records=[],
            discarded_reason=None
        )

    def to_dict(self):
        return asdict(self)


class MetaInfoWriter:
    def __init__(self, id: str, json_file_path: str):
        self.id = id
        self.json_file_path = json_file_path

    def _load_json(self):
        try:
            # check if file exists
            if not os.path.exists(self.json_file_path):
                return MetaInfo.default(self.id).to_dict()
            with open(self.json_file_path, "r") as f:
                json_str = f.read()
                json_obj = json.loads(json_str)
        except Exception as e:
            print(f"Failed to load {self.json_file_path}. {e}")
            raise e
        return json_obj

    def write_song_info(self, song_info: SongInfo):
        json_obj = self._load_json()
        json_obj["song_info"] = song_info.to_dict()

        try:
            with open(self.json_file_path, "w") as f:
                f.write(json.dumps(json_obj))
        except Exception as e:
            print(f"Failed to write {self.json_file_path}. {e}")
            raise e

    def file_exists(self, file_key: str) -> bool:
        json_obj = self._load_json()
        return any([f['key'] == file_key for f in json_obj['file_details']])

    def append_file_detail(self, file_detail: FileDetail):
        json_obj = self._load_json()
        json_obj["file_details"].append(file_detail.to_dict())
        json_obj["error"] = None

        try:
            with open(self.json_file_path, "w") as f:
                f.write(json.dumps(json_obj))
        except Exception as e:
            print(f"Failed to write {self.json_file_path}. {e}")
            raise e

    def append_process_record(self, process_record: ProcessRecord):
        json_obj = self._load_json()
        json_obj["process_records"].append(process_record.to_dict())
        json_obj["error"] = None

        try:
            with open(self.json_file_path, "w") as f:
                f.write(json.dumps(json_obj))
        except Exception as e:
            print(f"Failed to write {self.json_file_path}. {e}")
            raise e

    def discard(self, reason: str):
        json_obj = self._load_json()
        json_obj["discarded_reason"] = reason

        try:
            with open(self.json_file_path, "w") as f:
                f.write(json.dumps(json_obj))
        except Exception as e:
            print(f"Failed to write {self.json_file_path}. {e}")
            raise e

    def error(self, error: str):
        json_obj = self._load_json()
        json_obj["error"] = error

        try:
            with open(self.json_file_path, "w") as f:
                f.write(json.dumps(json_obj))
        except Exception as e:
            print(f"Failed to write {self.json_file_path}. {e}")
            raise e


class FileSaver(ABC):
    @abstractmethod
    def save_from_local_path(self, local_path: str) -> FileDetail:
        pass

    @abstractmethod
    def save_from_string(self, string: str) -> FileDetail:
        pass

    @abstractmethod
    def save_from_ndarray(self, ndarray: np.ndarray) -> FileDetail:
        pass


class LocalFileSaver(FileSaver):
    def __init__(self, base_dir_path: str, file_key: str):
        self.location = 'local'
        self.base_dir_path = base_dir_path
        self.file_key = file_key
        self.file_path = f"{self.base_dir_path}/{self.file_key}"

    def save_from_local_path(self, local_path: str) -> FileDetail:
        # Copy file to base dir
        try:
            subprocess.run(
                ['cp', local_path, self.file_path])
        except Exception as e:
            print(f"Failed to copy {local_path} to {self.base_dir_path}. {e}")
            raise e

        return FileDetail(
            location=self.location,
            key=self.file_key,
        )

    def save_from_string(self, string: str) -> FileDetail:
        # Save string to file
        try:
            with open(self.file_path, "w") as f:
                f.write(string)
        except Exception as e:
            print(f"Failed to write string to {self.base_dir_path}. {e}")
            raise e

        return FileDetail(
            location=self.location,
            key=self.file_key,
        )

    def save_from_ndarray(self, ndarray: np.ndarray) -> FileDetail:
        # Save ndarray to file
        try:
            with open(self.file_path, "w") as f:
                string = json.dumps(ndarray.tolist())
                f.write(string)
        except Exception as e:
            print(f"Failed to write ndarray to {self.base_dir_path}. {e}")
            raise e

        return FileDetail(
            location=self.location,
            key=self.file_key,
        )


class ProcessRunner(ABC):
    @abstractmethod
    def run(self, file_savers: List[FileSaver], inputs: List[Any]) -> Optional[FileDetail]:
        pass


class DownloadYoutubeProcessRunner(ProcessRunner):
    def __init__(self, temp_dir_path: str, base_dir_path: str, meta_info_writer: MetaInfoWriter):
        self.temp_dir_path = temp_dir_path
        self.base_dir_path = base_dir_path
        self.meta_info_writer = meta_info_writer

        # Run below on your local and paste the output
        # !ytmusicapi oauth
        with open('/tmp/youtube-oauth.json', 'w') as file:
            file.write('{"access_token": "ya29.a0AfB_byARMFsW2o9zMfN8QVvtaH1A7qUntxHtsVK37V57dwq_eD4L-DKUNxe5OrqRCHJLfMRedyjn_u0Gfx9F6l3z7vPBvfvlJ1AFHF0y7BzPLkauSd3jB_92zi6QLKkoNcq1C6_fe-3qfwCxDK1qUp7JzWRmJqzD8Vkm-OIpJ-Y7VBSNaCgYKATwSARASFQHGX2MiSqq-nWZvNsIJM4HNY0OmLw0183", "expires_in": 64238, "refresh_token": "1//0eWOvEBrb84fjCgYIARAAGA4SNwF-L9IrkaKX15-wybnc6tX48PnpD3DlhTbFFi-FipgYrC5g4g3l2HG2hDUVa9oVDLey56p61cg", "scope": "https://www.googleapis.com/auth/youtube", "token_type": "Bearer", "expires_at": 1699735372}')
        self.ytmusic = YTMusic("/tmp/youtube-oauth.json")

    def run(self, file_savers: List[FileSaver], inputs: List[str]) -> Optional[FileDetail]:
        id = inputs[0]
        temp_file_path = f"{self.temp_dir_path}/{id}.wav"
        # Download from Youtube
        try:
            subprocess.run(['yt-dlp', "-x", "--audio-format", "wav",
                           f"https://www.youtube.com/watch?v={id}", "-o", f"{temp_file_path}"])
        except Exception as e:
            print(f"Failed to download {id} from Youtube. {e}")
            raise e

        # Get song info from Youtube and write to meta info
        try:
            song = self.ytmusic.get_song(id)
            title = song['videoDetails']['title']
            artist = song['videoDetails']['author']
            self.meta_info_writer.write_song_info(
                SongInfo(title=title, artist=artist))
        except Exception as e:
            print(f"Failed to get song info from Youtube. {e}")
            raise e

        # Check sound length and if it is too long (over 10 minutes), return None
        try:
            y, sr = librosa.load(temp_file_path)
            if librosa.get_duration(y=y, sr=sr) > 600:
                # Remove temp file
                subprocess.run(['rm', temp_file_path])
                # disard this id
                self.meta_info_writer.discard(
                    "Sound length is too long (over 10 minutes)")
                return None
        except Exception as e:
            print(f"Failed to load {temp_file_path}. {e}")
            raise e

        file_saver = file_savers[0]
        file_detail = file_saver.save_from_local_path(temp_file_path)
        # remove temp file
        subprocess.run(['rm', temp_file_path])
        return file_detail


class DescribeAttributesProcessRunner(ProcessRunner):
    def __init__(self, temp_dir_path: str, base_dir_path: str):
        self.temp_dir_path = temp_dir_path
        self.base_dir_path = base_dir_path

    def run(self, file_savers: List[FileSaver], inputs: List[FileDetail]) -> FileDetail:
        wav_file = inputs[0]
        wav_file_path = f"{self.base_dir_path}/{wav_file.key}"
        try:
            audio = es.MonoLoader(filename=wav_file_path).compute()
            key_extractor = es.KeyExtractor()
            key, scale, strength = key_extractor(audio)
            tempo_extractor = es.RhythmExtractor2013(method="multifeature")
            tempo, beats, beats_confidence, _, beats_loudness = tempo_extractor(
                audio)
        except Exception as e:
            print(f"Failed to extract attributes from {wav_file.key}. {e}")
            raise e

        # Save as json file
        json_str = json.dumps({
            "tempo": tempo,
            "key": key,
            "scale": scale,
        })
        file_saver = file_savers[0]
        return file_saver.save_from_string(json_str)


class CutoutChorusProcessRunner(ProcessRunner):
    def __init__(self, temp_dir_path: str, base_dir_path: str):
        self.temp_dir_path = temp_dir_path
        self.base_dir_path = base_dir_path

    def run(self, file_savers: List[FileSaver], inputs: List[FileDetail]) -> FileDetail:
        wav_file = inputs[0]
        wav_file_path = f"{self.base_dir_path}/{wav_file.key}"
        temp_file_path = f"{self.temp_dir_path}/chorus-{wav_file.key}"
        try:
            pychorus.find_and_output_chorus(wav_file_path, temp_file_path)
        except Exception as e:
            print(f"Failed to extract chorus from {wav_file.key}. {e}")
            raise e

        file_saver = file_savers[0]
        file_detail = file_saver.save_from_local_path(temp_file_path)
        # remove temp file
        subprocess.run(['rm', temp_file_path])
        return file_detail


class NormalizeKeyToCProcessRunner(ProcessRunner):
    def __init__(self, temp_dir_path: str, base_dir_path: str):
        self.temp_dir_path = temp_dir_path
        self.base_dir_path = base_dir_path

    def run(self, file_savers: List[FileSaver], inputs: List[FileDetail]) -> FileDetail:
        wav_file = inputs[0]
        wav_file_path = f"{self.base_dir_path}/{wav_file.key}"
        temp_file_path = f"{self.temp_dir_path}/keytoc-{wav_file.key}"
        try:
            y, sr = librosa.load(wav_file_path, sr=None)
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            key = np.argmax(np.sum(chroma, axis=1))
            if key == 0:
                # No need to normalize
                subprocess.run(['cp', wav_file_path, temp_file_path])
            else:
                n_steps = 0 - key
                if n_steps > 6:
                    n_steps -= 12
                elif n_steps < -6:
                    n_steps += 12
                y_changed_key = librosa.effects.pitch_shift(
                    y, sr=sr, n_steps=float(n_steps))
                sf.write(temp_file_path, y_changed_key, sr)
        except Exception as e:
            print(f"Failed to normalize {wav_file.key} to C. {e}")
            raise e

        file_saver = file_savers[0]
        file_detail = file_saver.save_from_local_path(temp_file_path)
        # remove temp file
        subprocess.run(['rm', temp_file_path])
        return file_detail


class NormalizeTempoTo120ProcessRunner(ProcessRunner):
    def __init__(self, temp_dir_path: str, base_dir_path: str):
        self.temp_dir_path = temp_dir_path
        self.base_dir_path = base_dir_path

    def run(self, file_savers: List[FileSaver], inputs: List[FileDetail]) -> FileDetail:
        wav_file = inputs[0]
        wav_file_path = f"{self.base_dir_path}/{wav_file.key}"
        temp_file_path = f"{self.temp_dir_path}/tempoto120-{wav_file.key}"
        try:
            sr = 44100
            audio = es.MonoLoader(filename=wav_file_path,
                                  sampleRate=sr).compute()
            tempo_extractor = es.RhythmExtractor2013(method="multifeature")
            tempo, beats, beats_confidence, _, beats_loudness = tempo_extractor(
                audio)

            # normilize tempo to 120
            y_changed_tempo = librosa.effects.time_stretch(
                audio, rate=120 / tempo)

            sf.write(temp_file_path, y_changed_tempo, 44100)
        except Exception as e:
            print(f"Failed to normalize {wav_file.key} to 120. {e}")
            raise e

        file_saver = file_savers[0]
        file_detail = file_saver.save_from_local_path(temp_file_path)
        return file_detail


class TrackSeparationVocalProcessRunner(ProcessRunner):
    def __init__(self, temp_dir_path: str, base_dir_path: str):
        self.temp_dir_path = temp_dir_path
        self.base_dir_path = base_dir_path

    def run(self, file_savers: List[FileSaver], inputs: List[FileDetail]) -> FileDetail:
        wav_file = inputs[0]
        wav_file_path = f"{self.base_dir_path}/{wav_file.key}"
        filename = os.path.splitext(os.path.basename(wav_file.key))[0]
        temp_file_path = os.path.join(
            self.temp_dir_path, "mdx_extra", filename, f"vocals.wav")
        try:
            demucs.separate.main(
                ["-o", self.temp_dir_path, "--two-stems", "vocals", "-n", "mdx_extra", wav_file_path])
        except Exception as e:
            print(f"Failed to separate vocals from {wav_file.key}. {e}")
            raise e

        file_saver = file_savers[0]
        file_detail = file_saver.save_from_local_path(temp_file_path)
        # remove temp file
        subprocess.run(['rm', temp_file_path])
        return file_detail


class PreparationCropProcessRunner(ProcessRunner):
    def __init__(self, temp_dir_path: str, base_dir_path: str):
        self.temp_dir_path = temp_dir_path
        self.base_dir_path = base_dir_path

    def run(self, file_savers: List[FileSaver], inputs: List[FileDetail]) -> FileDetail:
        wav_file = inputs[0]
        wav_file_path = f"{self.base_dir_path}/{wav_file.key}"
        y, sr = librosa.load(wav_file_path, sr=None)

        desired_duration = 5  # seconds
        desired_length = int(desired_duration * sr)
        if len(y) > desired_length:
            # クリップ
            print(f"Cropped")
            y = y[:desired_length]
        elif len(y) < desired_length:
            # 0で埋める
            print(f"Zero filled")
            y = np.pad(y, (0, desired_length - len(y)))

        temp_file_path = f"{self.temp_dir_path}/crop-{wav_file.key}"
        sf.write(temp_file_path, y, sr)

        file_saver = file_savers[0]
        file_detail = file_saver.save_from_local_path(temp_file_path)
        # remove temp file
        subprocess.run(['rm', temp_file_path])
        return file_detail


class FeatureExtractionMelspectrogramProcessRunner(ProcessRunner):
    def __init__(self, temp_dir_path: str, base_dir_path: str):
        self.temp_dir_path = temp_dir_path
        self.base_dir_path = base_dir_path

    def run(self, file_savers: List[FileSaver], inputs: List[FileDetail]) -> FileDetail:
        wav_file = inputs[0]
        wav_file_path = f"{self.base_dir_path}/{wav_file.key}"
        try:
            y, sr = librosa.load(wav_file_path, sr=None)
            S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
            log_S = librosa.power_to_db(S, ref=np.max)
        except Exception as e:
            print(f"Failed to extract spectrogram from {wav_file.key}. {e}")
            raise e

        file_saver = file_savers[0]
        file_detail = file_saver.save_from_ndarray(log_S)
        return file_detail


class FeatureExtractionChromagramProcessRunner(ProcessRunner):
    def __init__(self, temp_dir_path: str, base_dir_path: str):
        self.temp_dir_path = temp_dir_path
        self.base_dir_path = base_dir_path

    def run(self, file_savers: List[FileSaver], inputs: List[FileDetail]) -> FileDetail:
        wav_file = inputs[0]
        wav_file_path = f"{self.base_dir_path}/{wav_file.key}"
        try:
            y, sr = librosa.load(wav_file_path, sr=None)
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        except Exception as e:
            print(f"Failed to extract chromagram from {wav_file.key}. {e}")
            raise e

        file_saver = file_savers[0]
        file_detail = file_saver.save_from_ndarray(chroma)
        return file_detail


class FeatureExtractionMfccProcessRunner(ProcessRunner):
    def __init__(self, temp_dir_path: str, base_dir_path: str):
        self.temp_dir_path = temp_dir_path
        self.base_dir_path = base_dir_path

    def run(self, file_savers: List[FileSaver], inputs: List[FileDetail]) -> FileDetail:
        wav_file = inputs[0]
        wav_file_path = f"{self.base_dir_path}/{wav_file.key}"
        try:
            y, sr = librosa.load(wav_file_path, sr=None)
            mfcc = librosa.feature.mfcc(y=y, sr=sr)
        except Exception as e:
            print(f"Failed to extract mfcc from {wav_file.key}. {e}")
            raise e

        file_saver = file_savers[0]
        file_detail = file_saver.save_from_ndarray(mfcc)
        return file_detail


def main():
    workflow_definition = WorkflowDefinition(
        id="workflow",
        start_process_id="youtube"
    )

    process_category_definitions = [
        ProcessCategoryDefinition(
            id="download",
        ),
        ProcessCategoryDefinition(
            id="describe",
        ),
        ProcessCategoryDefinition(
            id="cutout",
        ),
        ProcessCategoryDefinition(
            id="normalize",
        ),
        ProcessCategoryDefinition(
            id="trackseparation",
        ),
        ProcessCategoryDefinition(
            id="preparation",
        ),
        ProcessCategoryDefinition(
            id="featureextraction",
        ),
    ]

    process_definitions = [
        # download
        ProcessDefinition(
            id="youtube",
            process_category_id="download",
            append_id_as_suffix=True,
            input_data_types=[Datatype("id")],
            output_data_types=[Datatype("wav")],
        ),
        # describe
        ProcessDefinition(
            id="attributes",
            process_category_id="describe",
            input_data_types=[Datatype("wav")],
            output_data_types=[Datatype("json")],
        ),
        # cutout
        ProcessDefinition(
            id="chorus",
            process_category_id="cutout",
            input_data_types=[Datatype("wav")],
            output_data_types=[Datatype("wav")],
        ),
        # normalize
        ProcessDefinition(
            id="keytoc",
            process_category_id="normalize",
            input_data_types=[Datatype("wav")],
            output_data_types=[Datatype("wav")],
        ),
        ProcessDefinition(
            id="tempoto120",
            process_category_id="normalize",
            input_data_types=[Datatype("wav")],
            output_data_types=[Datatype("wav")],
        ),
        # trackseparation
        ProcessDefinition(
            id="vocals",
            process_category_id="trackseparation",
            input_data_types=[Datatype("wav")],
            output_data_types=[Datatype("wav")],
        ),
        # preparation
        ProcessDefinition(
            id="crop",
            process_category_id="preparation",
            input_data_types=[Datatype("wav")],
            output_data_types=[Datatype("wav")],
        ),
        # featureextraction
        ProcessDefinition(
            id="melspectrogram",
            process_category_id="featureextraction",
            input_data_types=[Datatype("wav")],
            output_data_types=[
                Datatype("json", sub_type_name="melspectrogram")],

        ),
        ProcessDefinition(
            id="chromagram",
            process_category_id="featureextraction",
            input_data_types=[Datatype("wav")],
            output_data_types=[Datatype("json", sub_type_name="chromagram")],
        ),
        ProcessDefinition(
            id="mfcc",
            process_category_id="featureextraction",
            input_data_types=[Datatype("wav")],
            output_data_types=[Datatype("json", sub_type_name="mfcc")],
        ),

    ]

    transition_definitions = [
        TransitionDefinition(
            source_process_id="youtube",
            target_process_id="attributes",
            prerequisite_process_ids=[],
        ),
        TransitionDefinition(
            source_process_id="youtube",
            target_process_id="tempoto120",
            prerequisite_process_ids=[],
        ),
        TransitionDefinition(
            source_process_id="youtube",
            target_process_id="chorus",
            prerequisite_process_ids=[],
        ),
        TransitionDefinition(
            source_process_id="chorus",
            target_process_id="tempoto120",
            prerequisite_process_ids=[],
        ),
        TransitionDefinition(
            source_process_id="tempoto120",
            target_process_id="keytoc",
            prerequisite_process_ids=[],
        ),
        TransitionDefinition(
            source_process_id="tempoto120",
            target_process_id="vocals",
            prerequisite_process_ids=["chorus"],
        ),
        TransitionDefinition(
            source_process_id="keytoc",
            target_process_id="vocals",
            prerequisite_process_ids=["chorus"],
        ),
        TransitionDefinition(
            source_process_id="vocals",
            target_process_id="crop",
            prerequisite_process_ids=[],
        ),
        TransitionDefinition(
            source_process_id="crop",
            target_process_id="melspectrogram",
            prerequisite_process_ids=[],
        ),
        TransitionDefinition(
            source_process_id="crop",
            target_process_id="chromagram",
            prerequisite_process_ids=[],
        ),
        TransitionDefinition(
            source_process_id="crop",
            target_process_id="mfcc",
            prerequisite_process_ids=[],
        ),
    ]

    # ワークフローの検証
    validity = check_workflow_validity(workflow_definition.start_process_id, {
                                       p.id: p for p in process_definitions}, transition_definitions)
    if validity is not None:
        print(validity)
        return

    mermaid_diagram = generate_mermaid_diagram(
        workflow_definition,
        process_category_definitions,
        process_definitions, transition_definitions)
    print(mermaid_diagram)

    # ワークフローの実行
    git_repository_root_dir_path = subprocess.run(
        ['git', 'rev-parse', '--show-toplevel'], capture_output=True).stdout.decode('utf-8').strip()
    data_dir_path = os.path.join(git_repository_root_dir_path, "data")
    file_dir_path = os.path.join(data_dir_path, "file")
    meta_info_base_dir_path = os.path.join(data_dir_path, "meta_info")
    os.makedirs(file_dir_path, exist_ok=True)
    os.makedirs(meta_info_base_dir_path, exist_ok=True)

    ids = [
        "3zh9Wb1KuW8",  # 死ぬのがいいわ
        "pKfg-khvlfs",  # แม่ของลูก - ผาขาว (メーコンルー)
    ]

    def process_by_id(id):
        print(f"Start {id}...")
        start_process = [p for p in process_definitions if p.id ==
                         workflow_definition.start_process_id][0]

        meta_info_writer = MetaInfoWriter(
            id=id, json_file_path=f"{meta_info_base_dir_path}/{id}.json")

        # ワークフローの実行

        def run_process(current_process_definition: ProcessDefinition, previous_outputs: List[FileDetail], process_stack: List[ProcessDefinition]) -> None:
            process_category_definition = [
                c for c in process_category_definitions if c.id == current_process_definition.process_category_id][0]
            process_name = f"{process_category_definition.id}_{current_process_definition.id}"
            appended_process_stack = process_stack + \
                [current_process_definition]
            appended_file_suffix_stack = [
                p.suffix(id) for p in appended_process_stack]
            file_name = '-'.join(appended_file_suffix_stack)
            file_keys = [
                f"{file_name}.{dt.name}" for dt in current_process_definition.output_data_types]
            next_process_definitions = []
            for t in transition_definitions:
                if t.source_process_id == current_process_definition.id:
                    process_stack_ids = [
                        ps.id for ps in appended_process_stack]
                    if len(t.prerequisite_process_ids) == 0 or all([pid in process_stack_ids for pid in t.prerequisite_process_ids]):
                        target_process = [
                            p for p in process_definitions if p.id == t.target_process_id][0]
                        next_process_definitions.append(target_process)
                    else:
                        print(
                            f"- Skip transition {t} because prerequisite processes are not satisfied. {process_stack_ids} vs {t.prerequisite_process_ids}")

            def run_next_processes(output: Optional[FileDetail]):
                for next_process_definition in next_process_definitions:
                    run_process(next_process_definition, [
                                output if output is not None else FileDetail(
                                    location='local',
                                    key=f"{file_name}.{dt.name}",
                                ) for dt in current_process_definition.output_data_types], appended_process_stack)

            # すでにファイルが存在する場合はスキップ
            if all([meta_info_writer.file_exists(file_key) for file_key in file_keys]):
                print(f"Skip {process_name} for {id}...")
                run_next_processes(None)
                return

            file_savers = [LocalFileSaver(
                base_dir_path=file_dir_path, file_key=file_key) for file_key in file_keys]

            process_runner: ProcessRunner
            if current_process_definition.id == 'youtube':
                temp_dir_path = f"/tmp/download-youtube"
                process_runner = DownloadYoutubeProcessRunner(
                    temp_dir_path=temp_dir_path, base_dir_path=file_dir_path, meta_info_writer=meta_info_writer)
            elif current_process_definition.id == 'attributes':
                temp_dir_path = f"/tmp/describe-attributes"
                process_runner = DescribeAttributesProcessRunner(
                    temp_dir_path=temp_dir_path, base_dir_path=file_dir_path)
            elif current_process_definition.id == 'chorus':
                temp_dir_path = f"/tmp/cutout-chorus"
                process_runner = CutoutChorusProcessRunner(
                    temp_dir_path=temp_dir_path, base_dir_path=file_dir_path)
            elif current_process_definition.id == 'keytoc':
                temp_dir_path = f"/tmp/normalize-keytoc"
                process_runner = NormalizeKeyToCProcessRunner(
                    temp_dir_path=temp_dir_path, base_dir_path=file_dir_path)
            elif current_process_definition.id == 'tempoto120':
                temp_dir_path = f"/tmp/normalize-tempoto120"
                process_runner = NormalizeTempoTo120ProcessRunner(
                    temp_dir_path=temp_dir_path, base_dir_path=file_dir_path)
            elif current_process_definition.id == 'vocals':
                temp_dir_path = f"/tmp/trackseparation-vocals"
                process_runner = TrackSeparationVocalProcessRunner(
                    temp_dir_path=temp_dir_path, base_dir_path=file_dir_path)
            elif current_process_definition.id == 'crop':
                temp_dir_path = f"/tmp/preparation-crop"
                process_runner = PreparationCropProcessRunner(
                    temp_dir_path=temp_dir_path, base_dir_path=file_dir_path)
            elif current_process_definition.id == 'melspectrogram':
                temp_dir_path = f"/tmp/featureextraction-melspectrogram"
                process_runner = FeatureExtractionMelspectrogramProcessRunner(
                    temp_dir_path=temp_dir_path, base_dir_path=file_dir_path)
            elif current_process_definition.id == 'chromagram':
                temp_dir_path = f"/tmp/featureextraction-chromagram"
                process_runner = FeatureExtractionChromagramProcessRunner(
                    temp_dir_path=temp_dir_path, base_dir_path=file_dir_path)
            elif current_process_definition.id == 'mfcc':
                temp_dir_path = f"/tmp/featureextraction-mfcc"
                process_runner = FeatureExtractionMfccProcessRunner(
                    temp_dir_path=temp_dir_path, base_dir_path=file_dir_path)
            else:
                raise Exception(
                    f"Process {current_process_definition.id} is not supported.")
            os.makedirs(temp_dir_path, exist_ok=True)

            try:
                print(
                    f"Run {process_name} for {id}...")
                started_at = datetime.now()
                output = process_runner.run(
                    file_savers=[f for f in file_savers], inputs=previous_outputs)
                finished_at = datetime.now()
                meta_info_writer.append_file_detail(output)
                meta_info_writer.append_process_record(
                    ProcessRecord(
                        process_id=current_process_definition.id,
                        started_at=started_at.isoformat(),
                        finished_at=finished_at.isoformat(),
                        elapsed_secs=(finished_at - started_at).total_seconds()
                    )
                )
                print(
                    f"Finished {process_name} for {id} : elapsed {finished_at - started_at}"
                )
            except Exception as e:
                print(f"Failed to run {current_process_definition.id}. {e}")
                traceback.print_exc()
                meta_info_writer.error(
                    f"Failed to run {current_process_definition.id}. {e}")
                return

            if output is None:
                # Discarded
                return

            if len(next_process_definitions) == 0:
                # ワークフローの終了
                return

            run_next_processes(output)

        run_process(start_process, [id], [])

    for id in ids:
        process_by_id(id)


if __name__ == "__main__":
    main()
