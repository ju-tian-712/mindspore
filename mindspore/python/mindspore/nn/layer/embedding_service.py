# Copyright 2024 Huawei Technologies Co., Ltd
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
"""embedding service"""
import json
import os
import math

from mindspore.nn.layer.embedding_service_layer import ESInitLayer
from mindspore.common.initializer import Uniform, TruncatedNormal, Constant
from mindspore.nn.layer.embedding_service_layer import ESEmbeddingTableImport, ESEmbeddingTableExport, \
    ESEmbeddingCKPTImport, ESEmbeddingCKPTExport

_INT32_MAX_VALUE = 2147483647


class CounterFilter:
    """ Counter filter for embedding table. """
    def __init__(self, filter_freq, default_key_or_value, default_key=None, default_value=None):
        self.filter_freq = filter_freq
        self.default_key = default_key
        self.default_value = default_value
        self.default_key_or_value = default_key_or_value
        print("CounterFilter params is: ", self.filter_freq, self.default_key, self.default_value,
              self.default_key_or_value, flush=True)


class EmbeddingVariableOption:
    """ option for embedding service table. """
    def __init__(self, filter_option=None,
                 evict_option=None,
                 storage_option=None,
                 feature_freezing_option=None,
                 communication_option=None):
        self.filter_option = filter_option
        self.evict_option = evict_option
        self.storage_option = storage_option
        self.feature_freezing_option = feature_freezing_option
        self.communication_option = communication_option


class EsInitializer:
    """Initializer for embedding service table."""
    def __init__(self, initializer_mode, min=-0.01, max=0.01, constant_value=1.0, mu=0.0, sigma=1.0, seed=0):
        self.initializer_mode = initializer_mode
        self.min = min
        self.max = max
        self.constant_value = constant_value
        self.mu = mu
        self.sigma = sigma
        self.seed = seed


class EsOptimizer:
    """Optimizer for embedding service table."""
    def __init__(self, name, initial_accumulator_value=0., ms=0., mom=0.):
        self.name = name
        self.initial_accumulator_value = initial_accumulator_value
        self.ms = ms
        self.mom = mom


def check_common_init_params(name, init_vocabulary_size, embedding_dim):
    """
    Check init params.
    """
    if (name is None) or (init_vocabulary_size is None) or (embedding_dim is None):
        raise ValueError("table name, init_vocabulary_size and embedding_dim can not be None.")
    if not isinstance(name, str):
        raise TypeError("embedding table name must be string.")
    if (not isinstance(init_vocabulary_size, int)) or (not isinstance(embedding_dim, int)):
        raise ValueError("init_vocabulary_size and embedding_dim must be int.")
    if init_vocabulary_size < 0:
        raise ValueError("init_vocabulary_size can not be smaller than zero.")
    if embedding_dim <= 0:
        raise ValueError("embedding_dim must be greater than zero.")


class EmbeddingService:
    """
    EmbeddingService
    """
    def __init__(self):
        """
        Init EmbeddingService
        """
        env_dist = os.environ
        es_cluster_config = env_dist.get("ESCLUSTER_CONFIG_PATH")
        if es_cluster_config is None:
            raise ValueError("EsClusterConfig env is null.")
        self._server_ip_to_ps_num = {}
        with open(es_cluster_config, encoding='utf-8') as a:
            es_cluster_config_json = json.load(a)
            self._es_cluster_conf = json.dumps(es_cluster_config_json)
            self._ps_num = int(es_cluster_config_json["psNum"])
            self._ps_ids = []
            self._ps_ids_list = es_cluster_config_json["psCluster"]
            for each_ps in self._ps_ids_list:
                self._server_ip_to_ps_num[each_ps["ctrlPanel"]["ipaddr"]] = 0

            for each_ps in self._ps_ids_list:
                self._ps_ids.append(each_ps["id"])
                ctrl_panel = each_ps["ctrlPanel"]
                self._server_ip_to_ps_num[ctrl_panel["ipaddr"]] += 1

            for each_server_ps_num in self._server_ip_to_ps_num:
                if self._server_ip_to_ps_num[each_server_ps_num] > 4:
                    raise ValueError("PS num of one server can not exceed 4, please check config params.")
                if self._ps_num > 4:
                    raise ValueError("PS num of one server can not exceed 4, please check config params.")

        # storage each ps table's params
        self._table_to_embedding_dim = {}
        self._table_to_max_num = {}
        self._table_to_optimizer = {}
        self._table_to_slot_var_num = {}
        self._table_to_counter_filter = {}
        self._train_mode = True
        self._train_level = False
        self._optimizer = None
        self._init_table_flag = False

        self._small_table_name_list = []
        self._ps_table_count = 0
        self._table_name_to_id = {}
        self._table_id_to_name = {}
        self._table_id_to_initializer = {}

        self._ps_table_id_list = []
        # storage lookup: table_id list, lookup result list, lookup key list
        self._ps_lookup_index = 0
        # storage all inited table names
        self._table_name_has_init = []
        # only storage all inited PS table names
        self._ps_table_name_list = []
        # now only use for adagrad accum
        self._ps_table_id_to_optimizer_params = {}

        # use for counter filter
        self._table_use_counter_filter = {}
        self._use_counter_filter = False

    def embedding_init(self, name, init_vocabulary_size, embedding_dim, max_feature_count,
                       initializer=Uniform(scale=0.01), ev_option=None, optimizer=None, optimizer_param=None,
                       mode="train"):
        """
        Init embedding
        :param name: big table name
        :param init_vocabulary_size: vocab size
        :param embedding_dim: embedding dim
        :param max_feature_count: max feature count
        :param initializer: mindspore common initializer
        :param ev_option: output of embedding_variable_option
        :param optimizer: optimizer
        :param optimizer_param: optimizer param
        :param mode: mode, train or predict
        :return: table_id_dict, es_initializer_dict, es_filter_dict
        """
        check_common_init_params(name=name, init_vocabulary_size=init_vocabulary_size, embedding_dim=embedding_dim)
        table_id = self._check_and_update_ps_init_params(name=name, init_vocabulary_size=init_vocabulary_size,
                                                         max_feature_count=max_feature_count, ev_option=ev_option)
        self._ps_lookup_index = self._ps_table_count
        self._table_to_embedding_dim[table_id] = embedding_dim
        self._table_to_max_num[table_id] = max_feature_count
        # storage the table id for embedding PS table
        self._ps_table_id_list.append(table_id)
        self._ps_table_name_list.append(name)

        if len(self._ps_table_id_list) > 10:
            raise ValueError("Now only 10 PS embedding tables can be init.")
        bucket_size = math.ceil(init_vocabulary_size / self._ps_num)
        if optimizer is None:
            self._train_mode = False
            self._table_to_slot_var_num[table_id] = 0
        else:
            self._check_ps_opt_and_initializer(optimizer=optimizer, initializer=initializer, table_id=table_id)
            self._optimizer = optimizer
            self._table_to_optimizer[table_id] = self._optimizer
            self._ps_table_id_to_optimizer_params[table_id] = []
            self._update_optimizer_slot_var_num(table_id=table_id)
            # new train or continue train from a checkpoint
            if initializer is not None:
                self._train_level = True
        filter_mode = self._init_counter_filter(table_id, ev_option)
        self._init_optimizer_mode_and_params(table_id, optimizer_param)
        es_init_layer = ESInitLayer(self._ps_num, self._ps_ids, self._train_mode, self._train_level, table_id,
                                    bucket_size, embedding_dim, self._table_to_slot_var_num.get(table_id),
                                    self._table_id_to_initializer.get(table_id), filter_mode, optimizer,
                                    self._ps_table_id_to_optimizer_params.get(table_id), max_feature_count, mode)
        es_init_layer()
        return self._table_name_to_id, self._table_id_to_initializer, self._table_to_counter_filter

    def counter_filter(self, filter_freq, default_key=None, default_value=None):
        """
        Set filter_option
        :param filter_freq: filter freq
        :param default_key: default key
        :param default_value: default value
        :return: CounterFilter obj
        """
        if not isinstance(filter_freq, int):
            raise TypeError("filter_freq must be int, please check.")
        if filter_freq < 0:
            raise ValueError("filter_freq must can not be smaller than 0.")
        if (default_key is None) and (default_value is None):
            raise ValueError("default_key and default_value can not be both None.")
        if (default_key is not None) and (default_value is not None):
            raise ValueError("default_key and default_value can not be both set.")
        if default_key is None and (not isinstance(default_value, (int, float))):
            raise TypeError("When default_value is not None, it must be float or int, please check.")
        if default_value is None and (not isinstance(default_key, int)):
            raise TypeError("When default_key is not None, it must be int, please check.")
        self._use_counter_filter = True
        if default_key is None:
            return CounterFilter(filter_freq=filter_freq, default_key_or_value=False,
                                 default_key=default_key, default_value=default_value)
        return CounterFilter(filter_freq=filter_freq, default_key_or_value=True,
                             default_key=default_key, default_value=default_value)

    def embedding_variable_option(self, filter_option=None, evict_option=None, storage_option=None,
                                  feature_freezing_option=None, communication_option=None):
        """
        Set embedding variable option
        :param filter_option: filter policy, is the output of counter_filter
        :param evict_option: not support
        :param storage_option: not support
        :param feature_freezing_option: not support
        :param communication_option: not support
        :return: EmbeddingVariableOption obj
        """
        if filter_option is None:
            raise ValueError("Now filter_option can't be None.")
        if not isinstance(filter_option, CounterFilter):
            raise TypeError("If filter_option isn't None, it must be CounterFilter type.")
        self._use_counter_filter = True
        return EmbeddingVariableOption(filter_option=filter_option, evict_option=evict_option,
                                       storage_option=storage_option, feature_freezing_option=feature_freezing_option,
                                       communication_option=communication_option)

    def embedding_ckpt_export(self, file_path):
        """
        Export big table ckpt
        :param file_path: the file path to storage ckpt ret
        :return:
        """
        embedding_dim_list = []
        value_total_len_list = []
        steps_to_live_list = []
        for table_id in self._ps_table_id_list:
            embedding_dim_list.append(self._table_to_embedding_dim.get(table_id))
            value_total_len_list.append(self._table_to_embedding_dim.get(table_id) *
                                        (self._table_to_slot_var_num.get(table_id) + 1) + 2)
            steps_to_live_list.append(0)
        embedding_ckpt_export_layer = ESEmbeddingCKPTExport(embedding_dim_list, value_total_len_list,
                                                            self._ps_table_name_list, self._ps_table_id_list,
                                                            file_path, steps_to_live_list)
        embedding_ckpt_export_layer()

    def embedding_table_export(self, file_path):
        """
        Export big table embedding
        :param file_path: the file path to storage embedding ret
        :return:
        """
        embedding_dim_list = []
        steps_to_live_list = []
        for table_id in self._ps_table_id_list:
            embedding_dim_list.append(self._table_to_embedding_dim.get(table_id))
            steps_to_live_list.append(0)

        embedding_table_export_layer = ESEmbeddingTableExport(embedding_dim_list, embedding_dim_list,
                                                              self._ps_table_name_list, self._ps_table_id_list,
                                                              file_path, steps_to_live_list)
        embedding_table_export_layer()

    def embedding_ckpt_import(self, file_path):
        """
        Import big table ckpt
        :param file_path: the file path to import ckpt ret
        :return:
        """
        embedding_dim_list = []
        value_total_len_list = []
        for table_id in self._ps_table_id_list:
            embedding_dim_list.append(self._table_to_embedding_dim.get(table_id))
            value_total_len_list.append(self._table_to_embedding_dim.get(table_id) *
                                        (self._table_to_slot_var_num.get(table_id) + 1) + 2)

        embedding_ckpt_export_layer = ESEmbeddingCKPTImport(embedding_dim_list, value_total_len_list,
                                                            self._ps_table_name_list, self._ps_table_id_list,
                                                            file_path)
        embedding_ckpt_export_layer()

    def embedding_table_import(self, file_path):
        """
        Import big table embedding
        :param file_path: the file path to import embedding ret
        :return:
        """
        embedding_dim_list = []
        for table_id in self._ps_table_id_list:
            embedding_dim_list.append(self._table_to_embedding_dim.get(table_id))
        embedding_table_export_layer = ESEmbeddingTableImport(embedding_dim_list, embedding_dim_list,
                                                              self._ps_table_name_list, self._ps_table_id_list,
                                                              file_path)
        embedding_table_export_layer()

    def _check_and_update_ps_init_params(self, name, init_vocabulary_size, max_feature_count, ev_option):
        """
        Check parameter server params and init table id
        """
        if max_feature_count is None:
            raise ValueError("For ps table, max_feature_count can not be None.")
        if (ev_option is not None) and (not isinstance(ev_option, EmbeddingVariableOption)):
            raise TypeError("For ps table, ev_option must be EmbeddingVariableOption type.")
        if not isinstance(max_feature_count, int):
            raise ValueError("For ps table, max_feature_count must be int.")
        if init_vocabulary_size >= _INT32_MAX_VALUE:
            raise ValueError("init_vocabulary_size exceeds int32 max value.")
        if max_feature_count <= 0:
            raise ValueError("For ps table, max_feature_count must be greater than zero.")
        if name not in self._table_name_has_init:
            table_id = self._ps_table_count
            self._table_name_to_id[name] = table_id
            self._table_id_to_name[table_id] = name
            self._ps_table_count += 1
            self._table_name_has_init.append(name)
        else:
            raise ValueError("This table has been initialized.")
        return table_id

    def _check_ps_opt_and_initializer(self, optimizer, initializer, table_id):
        """
        Check args of parameter server
        :param optimizer: the optimizer type, just support adam now
        :param initializer: mindspore common initializer
        :param table_id: table id
        :return:
        """
        if not optimizer in ["adam", "adagrad", "adamw"]:
            raise ValueError("optimizer should be one of adam, adagrad, adamw")
        if initializer is not None:
            if isinstance(initializer, EsInitializer):
                self._table_id_to_initializer[table_id] = initializer
            elif isinstance(initializer, TruncatedNormal):
                self._table_id_to_initializer[table_id] = \
                    EsInitializer(initializer_mode="truncated_normal", mu=initializer.mean,
                                  sigma=initializer.sigma, seed=initializer.seed[0])
            elif isinstance(initializer, Uniform):
                self._table_id_to_initializer[table_id] = \
                    EsInitializer(initializer_mode="random_uniform", min=-initializer.scale,
                                  max=initializer.scale, seed=initializer.seed[0])
            elif isinstance(initializer, Constant):
                self._table_id_to_initializer[table_id] = \
                    EsInitializer(initializer_mode="constant", constant_value=initializer.value)
            else:
                raise TypeError("initializer must be EsInitializer or mindspore initializer, and only support"
                                "Uniform, TruncatedNormal and Constant value.")

    def _update_optimizer_slot_var_num(self, table_id):
        """
        Update _table_to_slot_var_num by diff optimizer
        """
        # adam, adamw, rmsprop include m and v, 2 slots; adagrad include accumulator, 1 slot; sgd include 0 slot
        if self._optimizer == "adagrad":
            self._table_to_slot_var_num[table_id] = 1
        elif self._optimizer == "sgd":
            self._table_to_slot_var_num[table_id] = 0
        else:
            self._table_to_slot_var_num[table_id] = 2

    def _init_counter_filter(self, table_id, ev_option):
        """
        Init counter filter params
        """
        if (ev_option is not None) and (ev_option.filter_option is not None):
            filter_mode = "counter"
            self._table_to_counter_filter[table_id] = ev_option.filter_option
            self._table_use_counter_filter[table_id] = 1
        else:
            filter_mode = "no_filter"
            self._table_use_counter_filter[table_id] = 0
        return filter_mode

    def _init_optimizer_mode_and_params(self, table_id, optimizer_param):
        """
        Init _ps_table_id_to_optimizer_params by diff optimizer
        """
        optimizer = self._table_to_optimizer.get(table_id)
        if optimizer is None:
            return
        if optimizer == "adagrad":
            if optimizer_param is None or len(optimizer_param) != 1:
                self._ps_table_id_to_optimizer_params[table_id].extend(optimizer_param)
            else:
                raise ValueError("For adagrad optimizer, optimizer_param should have 1 param, "
                                 "initial_accumulator_value")
        if optimizer in ["adam", "adamw", "sgd"]:
            self._ps_table_id_to_optimizer_params[table_id].append(0.)
