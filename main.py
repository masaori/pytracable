from dataclasses import dataclass, asdict
import os
from typing import Optional, List
import json
from abc import ABC, abstractmethod
import subprocess
import traceback
# pip
import boto3
from ytmusicapi import YTMusic
import essentia.standard as es
import librosa
import numpy as np
import soundfile as sf
import gspread


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


@dataclass
class TransitionDefinition(EntityBase):
    source_process_id: str
    target_process_id: str

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
    data_type: Datatype

    def to_dict(self):
        return asdict(self)


@dataclass
class MetaInfo:
    id: str
    title: str
    artist: str
    files: List[FileDetail]
    discarded_reason: Optional[str] = None
    error: Optional[str] = None


class MetaInfoWriter:
    def __init__(self, id: str, json_file_path: str):
        self.id = id
        self.json_file_path = json_file_path

    def _load_json(self):
        try:
            # check if file exists
            if not os.path.exists(self.json_file_path):
                return {
                    "id": self.id,
                    "title": None,
                    "artist": None,
                    "files": [],
                    "discarded_reason": None,
                    "error": None,
                }
            with open(self.json_file_path, "r") as f:
                json_str = f.read()
                json_obj = json.loads(json_str)
        except Exception as e:
            print(f"Failed to load {self.json_file_path}. {e}")
            raise e
        return json_obj

    def write_song_info(self, title: str, artist: str):
        json_obj = self._load_json()
        json_obj["title"] = title
        json_obj["artist"] = artist

        try:
            with open(self.json_file_path, "w") as f:
                f.write(json.dumps(json_obj))
        except Exception as e:
            print(f"Failed to write {self.json_file_path}. {e}")
            raise e

    def file_exists(self, file_detail: FileDetail) -> bool:
        json_obj = self._load_json()
        return any([f['key'] == file_detail.key for f in json_obj['files']])

    def append_file(self, file_detail: FileDetail):
        json_obj = self._load_json()
        json_obj["files"].append(file_detail.to_dict())
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
    def save_from_local_path(self, local_path: str, data_type: Datatype) -> FileDetail:
        pass

    @abstractmethod
    def save_from_string(self, string: str, data_type: Datatype) -> FileDetail:
        pass


class LocalFileSaver:
    def __init__(self, base_dir_path: str, file_key: str):
        self.location = 'local'
        self.base_dir_path = base_dir_path
        self.file_key = file_key
        self.file_path = f"{self.base_dir_path}/{self.file_key}"

    def save_from_local_path(self, local_path: str, data_type: Datatype) -> FileDetail:
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
            data_type=data_type
        )

    def save_from_string(self, string: str, data_type: Datatype) -> FileDetail:
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
            data_type=data_type
        )


class ProcessRunner(ABC):
    @abstractmethod
    def run(self, file_savers: List[FileSaver], inputs: List[any]) -> Optional[FileDetail]:
        pass


class DownloadYoutubeProcessRunner(ProcessRunner):
    def __init__(self, temp_dir_path: str, meta_info_writer: MetaInfoWriter):
        self.temp_dir_path = temp_dir_path
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
                title=title, artist=artist)
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
        file_detail = file_saver.save_from_local_path(
            temp_file_path, Datatype("wav"))
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
        return file_saver.save_from_string(json_str, Datatype("json"))


# class CutoutChorusProcessRunner(ProcessRunner):


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
        # ProcessDefinition(
        #     id="backtracks",
        #     process_category_id="trackseparation",
        #     input_data_types=[Datatype("wav")],
        #     output_data_types=[Datatype("wav")],
        # ),
        # featureextraction
        ProcessDefinition(
            id="spectrogram",
            process_category_id="featureextraction",
            input_data_types=[Datatype("wav")],
            output_data_types=[Datatype("json", sub_type_name="spectrogram")],
        ),
        ProcessDefinition(
            id="melspectrogram",
            process_category_id="featureextraction",
            input_data_types=[Datatype("wav")],
            output_data_types=[
                Datatype("wav", sub_type_name="melspectrogram")],

        ),
        ProcessDefinition(
            id="chromagram",
            process_category_id="featureextraction",
            input_data_types=[Datatype("wav")],
            output_data_types=[Datatype("wav", sub_type_name="chromagram")],
        ),
    ]

    transition_definitions = [
        TransitionDefinition(
            source_process_id="youtube",
            target_process_id="attributes",
        ),
        # TransitionDefinition(
        #     source_process_id="youtube",
        #     target_process_id="keytoc",
        # ),
        # TransitionDefinition(
        #     source_process_id="youtube",
        #     target_process_id="tempoto120",
        # ),
        # TransitionDefinition(
        #     source_process_id="youtube",
        #     target_process_id="chorus",
        # ),
        # TransitionDefinition(
        #     source_process_id="chorus",
        #     target_process_id="keytoc",
        # ),
        # TransitionDefinition(
        #     source_process_id="chorus",
        #     target_process_id="tempoto120",
        # ),
        # TransitionDefinition(
        #     source_process_id="keytoc",
        #     target_process_id="tempoto120",
        # ),
        # TransitionDefinition(
        #     source_process_id="tempoto120",
        #     target_process_id="vocals",
        # ),
        # TransitionDefinition(
        #     source_process_id="vocals",
        #     target_process_id="spectrogram",
        # ),
        # TransitionDefinition(
        #     source_process_id="vocals",
        #     target_process_id="melspectrogram",
        # ),
        # TransitionDefinition(
        #     source_process_id="vocals",
        #     target_process_id="chromagram",
        # ),
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
    # メタ情報の保存先
    meta_info_base_dir_path = os.path.join(data_dir_path, "meta_info")
    os.makedirs(meta_info_base_dir_path, exist_ok=True)

    ids = [
        "3zh9Wb1KuW8",  # 死ぬのがいいわ
        "pKfg-khvlfs",  # แม่ของลูก - ผาขาว (メーコンルー)
    ]

    for id in ids:
        print(f"Start {id}...")
        start_process = [p for p in process_definitions if p.id ==
                         workflow_definition.start_process_id][0]

        meta_info_writer = MetaInfoWriter(
            id=id, json_file_path=f"{meta_info_base_dir_path}/{id}.json")

        # ワークフローの実行

        def run_process(process_definition: ProcessDefinition, previous_outputs: List[FileDetail], file_suffix_stack: List[str]) -> None:
            process_category_definition = [
                c for c in process_category_definitions if c.id == process_definition.process_category_id][0]
            process_name = f"{process_category_definition.id}_{process_definition.id}"
            append_file_suffix = f"{id}_{process_definition.id}" if process_definition.append_id_as_suffix else f"{process_category_definition.id}_{process_definition.id}"
            appended_file_suffix_stack = file_suffix_stack + \
                [append_file_suffix]
            file_name = '-'.join(appended_file_suffix_stack)
            file_keys = [
                f"{file_name}.{dt.name}" for dt in process_definition.output_data_types]

            # すでにファイルが存在する場合はスキップ
            if all([meta_info_writer.file_exists(FileDetail(location='local', key=file_key, data_type=Datatype('wav'))) for file_key in file_keys]):
                print(f"Skip {process_name} for {id}...")
                return

            file_savers = [LocalFileSaver(
                base_dir_path=data_dir_path, file_key=file_key) for file_key in file_keys]

            if process_definition.id == 'youtube':
                temp_dir_path = f"/tmp/download-youtube"
                process_runner = DownloadYoutubeProcessRunner(
                    temp_dir_path=temp_dir_path, meta_info_writer=meta_info_writer)
            elif process_definition.id == 'attributes':
                temp_dir_path = f"/tmp/describe-attributes"
                process_runner = DescribeAttributesProcessRunner(
                    temp_dir_path=temp_dir_path, base_dir_path=data_dir_path)
            else:
                raise Exception(
                    f"Process {process_definition.id} is not supported.")
            os.makedirs(temp_dir_path, exist_ok=True)

            try:
                print(
                    f"Run {process_name} for {id}...")
                outputs = process_runner.run(
                    file_savers=file_savers, inputs=previous_outputs)
                meta_info_writer.append_file(outputs)
                print(
                    f"Finished {process_name} for {id}..."
                )
            except Exception as e:
                print(f"Failed to run {process_definition.id}. {e}")
                traceback.print_exc()
                meta_info_writer.error(
                    f"Failed to run {process_definition.id}. {e}")
                return

            if outputs is None:
                return

            next_process_definitions = [p for p in process_definitions if p.id in [
                t.target_process_id for t in transition_definitions if t.source_process_id == process_definition.id]]

            if len(next_process_definitions) == 0:
                # ワークフローの終了
                return

            for next_process_definition in next_process_definitions:
                run_process(next_process_definition, [
                            outputs], appended_file_suffix_stack)

        run_process(start_process, [id], [])


if __name__ == "__main__":
    main()
