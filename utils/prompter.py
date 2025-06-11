import json
import os.path as osp
from typing import Union


class Prompter(object):
    __slots__ = ("template", "_verbose")

    def __init__(self, template_name: str = "", verbose: bool = False):
        self._verbose = verbose
        if not template_name:
            template_name = "alpaca" # デフォルトのテンプレート
        
        # templatesフォルダへの正しいパスを設定
        # prompter.pyからの相対パスでtemplatesフォルダを探す
        current_dir = osp.dirname(osp.abspath(__file__))
        template_dir = osp.join(current_dir, "templates")
        file_name = osp.join(template_dir, f"{template_name}.json")

        if not osp.exists(file_name):
            raise ValueError(f"Can't find template file: {file_name}")
        with open(file_name, encoding='utf-8') as fp: # エンコーディングを指定
            self.template = json.load(fp)
        if self._verbose:
            print(
                f"Using prompt template {template_name}: {self.template['description']}"
            )

    def generate_prompt(
        self,
        instruction: str,
        input: Union[None, str] = None,
        label: Union[None, str] = None,
    ) -> str:
        # returns the full prompt from instruction and optional input
        # if a label (=response, =output) is provided, it's also appended.
        if input:
            res = self.template["prompt_input"].format(
                instruction=instruction, input=input
            )
        else:
            res = self.template["prompt_no_input"].format(
                instruction=instruction
            )
        if label:
            res = f"{res}{label}"
        if self._verbose:
            print(res)
        return res

    def get_response(self, output: str) -> str:
        # outputからresponse_split以降を抽出
        # outputがprompt+responseの形式になっていることを前提
        parts = output.split(self.template["response_split"])
        if len(parts) > 1:
            return parts[1].strip()
        return output.strip() # もしsplitが機能しない場合は元のoutputを返す
    
    def get_response_from_icl(self, input: str, output: str) -> str:
        # ICL (In-Context Learning) の場合、outputは入力と生成された応答を含む
        # そのため、入力部分を除去し、response_split以降を抽出する
        # ここではinputとoutputの文字列長で簡易的に処理しているが、
        # トークンレベルでの正確な処理が必要になる場合がある。
        # LLaVAではプロンプトの構造が複雑になるため、このメソッドの正確な動作は要確認。
        
        # 元のコードを保持しつつ、必要に応じて修正
        # LLaVAのプロンプト形式が `USER: <image>\n{instruction} ASSISTANT:` の場合、
        # `input`が画像パスで、その後のテキストを区別する。
        # ここでは元のコードの意図通り、文字列の長さでカットオフする。
        # ただし、`response_split`が複数回出現する可能性も考慮が必要。
        response_start_index = output.find(self.template["response_split"])
        if response_start_index != -1:
            return output[response_start_index + len(self.template["response_split"]):].strip()
        return output.strip()