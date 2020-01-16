
class PathDict:
    def __init__(self, init_dict={}):
        self.internal_dict = {}
        for key in list(init_dict.keys()):
            self.__setitem__(key, init_dict[key])

    def __getitem__(self, key):
        if type(key) != list:
            key = key.split("/")
        local_key = key[0]
        child_key = key[1:]

        try:
            local_node = self.internal_dict[local_key]
        except KeyError:
            print("Unknown key " + local_key)
            print(self.keys(recursive=True))
        if len(child_key) > 0:
            return local_node[child_key]
        else:
            return local_node

    def __setitem__(self, key, value):
        if type(key) is not list:
            key = key.split("/")
        local_key = key[0]
        child_key = key[1:]

        if isinstance(value, dict):
            converted_dict = PathDict()
            converted_dict.update(value)
            value = converted_dict

        if len(child_key) > 0:
            if local_key not in self.internal_dict:
                self.internal_dict[local_key]=PathDict()
            self.internal_dict[local_key][child_key] = value
        else:
            if local_key.endswith(":[]"):
                local_key_s = local_key.split(":")
                if len(local_key_s) != 2:
                    raise Exception("Invalid key format : '" + local_key + "'")
                local_key = local_key_s[0]
                if local_key not in self.internal_dict:
                    self.internal_dict[local_key] = []
                if not isinstance(self.internal_dict[local_key], list):
                    raise Exception("Trying to happend to a leaf that is not a list '"+local_key + "'")
                self.internal_dict[local_key].append(value)
            else:
                self.internal_dict[local_key] = value


    def __delitem__(self, key):
        if type(key) != list:
            key = key.split("/")
        local_key = key[0]
        child_key = key[1:]

        if len(child_key) > 0:
            del self.internal_dict[local_key][child_key]
        else:
            del self.internal_dict[local_key]

    def keys(self, leafs=False, recursive=False, node_path=None):
        local_keys = list(self.internal_dict.keys())

        keys = []
        for k in local_keys:
            if node_path is not None:
                full_key = node_path+"/"+k
            else:
                full_key = k
            if isinstance(self.internal_dict[k], PathDict):
                if recursive:
                    keys += self.internal_dict[k].keys(leafs=leafs, recursive=recursive, node_path=full_key)
                if not leafs:
                    keys.append(full_key)
            else:
                keys.append(full_key)
        return keys

    def values(self):
        return self.internal_dict.values()

    def update(self, other_dict):
        if isinstance(other_dict, PathDict):
            list_of_keys = other_dict.keys(leafs=True, recursive=True)
        else:
            list_of_keys = other_dict.keys()
        for key in list_of_keys:
            value = other_dict[key]
            if isinstance(value, dict):
                converted_dict = PathDict()
                converted_dict.update(value)
                value = converted_dict
            self.__setitem__(key, value)

    def __contains__(self, key):
        return key in self.keys(recursive=True)