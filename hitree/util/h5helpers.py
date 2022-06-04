from typing import Any

import h5py


def create_dataset_if_not_exists(ds_name: str, group: h5py.Group, **kwargs) -> h5py.Dataset:
    if ds_name in group.keys():
        return group[ds_name]
    else:
        return group.create_dataset(ds_name, **kwargs)


def create_group_if_not_exists(group_name: str, group: h5py.Group, **kwargs) -> h5py.Group:
    if group_name in group.keys():
        return group[group_name]
    else:
        return group.create_group(group_name, **kwargs)


def get_attribute_value_or_create_if_not_exists(attr_name: str, default_value: Any, group: h5py.Group, **kwargs) -> Any:
    if attr_name in group.attrs.keys():
        return group.attrs.get(attr_name)
    else:
        group.attrs.create(attr_name, data=default_value, **kwargs)
        return default_value
