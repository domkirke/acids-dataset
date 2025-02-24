import pytest
import numpy as np
import torch
from acids_dataset.fragments import AcidsFragment

@pytest.mark.parametrize("bformat", ["int16", "float32"])
# @pytest.mark.parametrize("output_type", ["numpy", "torch", "jax"])
@pytest.mark.parametrize("output_type", ["numpy", "torch"])
def test_acids_fragments(bformat, output_type):
    noise_in = torch.rand(1, 131072)
    metadata_int_in = torch.randperm(15)
    metadata_float_in = torch.randn(1, 128)

    if output_type == "numpy":
        noise_out = noise_in.numpy()
        metadata_int_out = metadata_int_in.numpy()
        metadata_float_out = metadata_float_in.numpy()
    if output_type == "torch":
        noise_out = noise_in
        metadata_int_out = metadata_int_in
        metadata_float_out = metadata_float_in
    fg = AcidsFragment(
        start_pos = 0., 
        bformat=bformat, 
        output_type=output_type
    )

    fg.put_audio("audio", noise_in.numpy(), bformat)
    fg.put_array("category", metadata_int_in.numpy())
    fg.put_array("profile", metadata_float_in.numpy())
    fragment_out = fg.get("audio")
    meta_int_out = fg.get_array("category")
    meta_float_out = fg.get_array("profile")
    if output_type == "numpy":
        assert isinstance(fragment_out, np.ndarray)
        assert not np.issubdtype(fragment_out.dtype, np.integer)
        assert np.allclose(fragment_out, noise_out, atol=1e-4)
    elif output_type == "torch":
        assert torch.is_tensor(fragment_out)
        # lame, but.... :')
        assert "int" not in str(fragment_out.dtype)
        assert torch.allclose(fragment_out, noise_out, atol=1e-4)
    assert meta_int_out.dtype == metadata_int_out.dtype
    assert meta_float_out.dtype == metadata_float_out.dtype
    


