# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Parse ast.Assign in construct function to node of SymbolTree."""
import ast

from mindspore.rewrite.parser import Parser
from mindspore.rewrite.symbol_tree import SymbolTree
from mindspore.rewrite.parser_register import reg_parser


class NameParser(Parser):
    """Parse ast.Name in construct function to node of SymbolTree."""
    def target(self):
        """Parse target type."""
        return ast.Name

    def process(self, stree: SymbolTree, node: ast.Name):
        """
        Parse ast.Name node.

        Args:
            stree ([SymbolTree]): Symbol Tree under parsing.
            node ([ast.Name]): An ast.Name node.

        Raises:
            TypeError: Name parser only supports parsing ast.Name type nodes.
        """
        if not isinstance(node, ast.Name):
            raise TypeError("Name parser only supports parsing ast.Name type nodes.")
        return node.id


class NumParser(Parser):
    """Parse ast.Num in construct function to node of SymbolTree."""
    def target(self):
        """Parse target type."""
        return ast.Num

    def process(self, stree: SymbolTree, node: ast.Num):
        """
        Parse ast.Num node.

        Args:
            stree ([SymbolTree]): Symbol Tree under parsing.
            node ([ast.Num]): An ast.Num node.

        Raises:
            TypeError: Num parser only supports parsing ast.Num type nodes.
        """
        if not isinstance(node, ast.Num):
            raise TypeError("Num parser only supports parsing ast.Num type nodes.")
        return node.n


class StrParser(Parser):
    """Parse ast.Str in construct function to node of SymbolTree."""
    def target(self):
        """Parse target type."""
        return ast.Str

    def process(self, stree: SymbolTree, node: ast.Str):
        """
        Parse ast.Str node.

        Args:
            stree ([SymbolTree]): Symbol Tree under parsing.
            node ([ast.Str]): An ast.Str node.

        Returns:
            The value of node.

        Raises:
            TypeError:Str parser only supports parsing ast.Str type nodes.
        """
        if not isinstance(node, ast.Str):
            raise TypeError("Str parser only supports parsing ast.Str type nodes.")
        return node.s

g_name_parser = reg_parser(NameParser())
g_num_parser = reg_parser(NumParser())
g_str_parser = reg_parser(StrParser())
