from pathlib import Path
import numpy as np
import numpy.testing
import pytest
from numpy.dtypes import StringDType

from davislib import ImageSetAccessor
from davislib.attribute import AttributeLevel


class TestImageSetProperties:
    def test_image_set_title(self, data_path):
        with ImageSetAccessor(data_path / 'SimpleImageSet') as images:
            assert images.title == 'Simple Image Set'

    def test_nbuffer(self, data_path):
        with ImageSetAccessor(data_path / 'SimpleImageSet') as images:
            assert images.dimensions.shape == (10,)
            assert images.dimensions.names == ('buffer',)

    def test_number_of_components(self, data_path):
        with ImageSetAccessor(data_path / 'SimpleImageSet') as images:
            assert len(images.components) == 1

    def test_component_properties(self, data_path):
        with ImageSetAccessor(data_path / 'SimpleImageSet') as images:
            assert list(images.components.keys()) == ['PIXEL']
            component = images.components['PIXEL']
            assert component.dimensions.shape == (10, 250, 2560)
            assert component.dimensions.names == ('buffer', 'y', 'x')
            assert component.dtype == np.uint16
            assert component.scale.slope == 1
            assert component.scale.offset == 0
            assert component.scale.unit == 'counts'
            assert component.scale.dtype == np.uint8


class TestListAttributes:
    def test_list_raw_attributes(self, data_path):
        with ImageSetAccessor(data_path / 'SimpleImageSet') as images:
            attrs = images.list_attributes(infer_types=False)
            assert len(attrs) == 114

    @pytest.mark.parametrize(
        'key,level,value',
        [
            pytest.param('Acq.Input.SpeedSelect', 'buffer', '0'),
            pytest.param('Acq.Input.StartTrigger', 'buffer', '0'),
            pytest.param('Acq.Status.RecordPost', 'buffer', '0'),
            pytest.param('CustomImageTag_Count', 'buffer', '0'),
            pytest.param('DevDataAlias0', 'buffer', 'Camera 1 : Exposure time'),
            pytest.param('DevDataAlias1', 'buffer', 'DyeLaser 1 : Wavelength'),
            pytest.param('DevDataAlias10', 'buffer', 'Timinig unit : TTL out'),
            pytest.param(
                'DevDataAlias2', 'buffer', 'Energy [Pulse 1, Head 1, Device 0]'
            ),
            pytest.param(
                'DevDataAlias3', 'buffer', 'Energy [Pulse 1, Head 1, Device 1]'
            ),
            pytest.param('DevDataAlias4', 'buffer', 'I/I 1 (Camera 1) : Delay'),
            pytest.param('DevDataAlias5', 'buffer', 'I/I 1 (Camera 1) : Gain'),
            pytest.param('DevDataAlias6', 'buffer', 'I/I 1 (Camera 1) : Gate'),
            pytest.param('DevDataAlias7', 'buffer', 'LightSource 1 : Power'),
            pytest.param('DevDataAlias8', 'buffer', 'LightSource 2 : Power'),
            pytest.param('DevDataAlias9', 'buffer', 'Reference time : Time 1'),
            pytest.param('DevDataChannel0', 'buffer', '0'),
            pytest.param('DevDataChannel1', 'buffer', '0'),
            pytest.param('DevDataChannel10', 'buffer', '0'),
            pytest.param('DevDataChannel2', 'buffer', '0'),
            pytest.param('DevDataChannel3', 'buffer', '0'),
            pytest.param('DevDataChannel4', 'buffer', '0'),
            pytest.param('DevDataChannel5', 'buffer', '0'),
            pytest.param('DevDataChannel6', 'buffer', '0'),
            pytest.param('DevDataChannel7', 'buffer', '0'),
            pytest.param('DevDataChannel8', 'buffer', '0'),
            pytest.param('DevDataChannel9', 'buffer', '0'),
            pytest.param('DevDataClass0', 'buffer', '1'),
            pytest.param('DevDataClass1', 'buffer', '1'),
            pytest.param('DevDataClass10', 'buffer', '1'),
            pytest.param('DevDataClass2', 'buffer', '0'),
            pytest.param('DevDataClass3', 'buffer', '0'),
            pytest.param('DevDataClass4', 'buffer', '1'),
            pytest.param('DevDataClass5', 'buffer', '1'),
            pytest.param('DevDataClass6', 'buffer', '1'),
            pytest.param('DevDataClass7', 'buffer', '1'),
            pytest.param('DevDataClass8', 'buffer', '1'),
            pytest.param('DevDataClass9', 'buffer', '1'),
            pytest.param(
                'DevDataName0',
                'buffer',
                'Camera.ExposureTime [Camera.ImagerSCMOS5MCLHS: 61008958]',
            ),
            pytest.param(
                'DevDataName1',
                'buffer',
                'DyeLaser.Wavelength [DyeLaser.SirahUsbDyeLaser: 24-08-19]',
            ),
            pytest.param(
                'DevDataName10',
                'buffer',
                'TimingUnit.TtlOut [TimingUnit.HSCv2: VZ23-1197]',
            ),
            pytest.param(
                'DevDataName2', 'buffer', 'Energy [Pulse 1, Head 1, Device 0]'
            ),
            pytest.param(
                'DevDataName3', 'buffer', 'Energy [Pulse 1, Head 1, Device 1]'
            ),
            pytest.param(
                'DevDataName4',
                'buffer',
                'IroDelayScanValue [ImageIntensifier.Iro10: VC23-0424] (Camera.ImagerSCMOS5MCLHS: 61008958)',
            ),
            pytest.param(
                'DevDataName5',
                'buffer',
                'IroGainScanValue [ImageIntensifier.Iro10: VC23-0424] (Camera.ImagerSCMOS5MCLHS: 61008958)',
            ),
            pytest.param(
                'DevDataName6',
                'buffer',
                'IroGateScanValue [ImageIntensifier.Iro10: VC23-0424] (Camera.ImagerSCMOS5MCLHS: 61008958)',
            ),
            pytest.param(
                'DevDataName7',
                'buffer',
                'LightPowerScanValue_Delay [LightSource.SinglePulseYAGLaser: SN_0001]',
            ),
            pytest.param(
                'DevDataName8',
                'buffer',
                'LightPowerScanValue_Delay [LightSource.SinglePulseYAGLaser: SN_0002]',
            ),
            pytest.param(
                'DevDataName9', 'buffer', 'Reference time 1 [Scanning.ReferenceTime: 0]'
            ),
            pytest.param('DevDataReference2', 'buffer', '2000'),
            pytest.param('DevDataReference3', 'buffer', '2000'),
            pytest.param('DevDataScaleI0', 'buffer', '1\n0\nµs\n'),
            pytest.param('DevDataScaleI1', 'buffer', '1\n0\nnm\n'),
            pytest.param('DevDataScaleI10', 'buffer', '1\n0\nµs\n'),
            pytest.param('DevDataScaleI2', 'buffer', '1\n0\ncounts\nEnergy'),
            pytest.param('DevDataScaleI3', 'buffer', '1\n0\ncounts\nEnergy'),
            pytest.param('DevDataScaleI4', 'buffer', '1\n0\nns\n'),
            pytest.param('DevDataScaleI5', 'buffer', '1\n0\n%\n'),
            pytest.param('DevDataScaleI6', 'buffer', '1\n0\nns\n'),
            pytest.param('DevDataScaleI7', 'buffer', '1\n0\n%\n'),
            pytest.param('DevDataScaleI8', 'buffer', '1\n0\n%\n'),
            pytest.param('DevDataScaleI9', 'buffer', '1\n0\nµs\n'),
            pytest.param('DevDataScaleX0', 'buffer', '1\n0\n\n'),
            pytest.param('DevDataScaleX1', 'buffer', '1\n0\n\n'),
            pytest.param('DevDataScaleX10', 'buffer', '1\n0\n\n'),
            pytest.param('DevDataScaleX2', 'buffer', '1\n0\nsample #\nSamples'),
            pytest.param('DevDataScaleX3', 'buffer', '1\n0\nsample #\nSamples'),
            pytest.param('DevDataScaleX4', 'buffer', '1\n0\n\n'),
            pytest.param('DevDataScaleX5', 'buffer', '1\n0\n\n'),
            pytest.param('DevDataScaleX6', 'buffer', '1\n0\n\n'),
            pytest.param('DevDataScaleX7', 'buffer', '1\n0\n\n'),
            pytest.param('DevDataScaleX8', 'buffer', '1\n0\n\n'),
            pytest.param('DevDataScaleX9', 'buffer', '1\n0\n\n'),
            pytest.param('DevDataSources', 'buffer', '11'),
            pytest.param(
                'DevDataTrace0', 'buffer', np.array([[9000.005]], dtype=np.float32)
            ),
            pytest.param(
                'DevDataTrace1', 'buffer', np.array([[628.2353]], dtype=np.float32)
            ),
            pytest.param(
                'DevDataTrace10', 'buffer', np.array([[0.0]], dtype=np.float32)
            ),
            pytest.param(
                'DevDataTrace2', 'buffer', np.array([[1557.0]], dtype=np.float32)
            ),
            pytest.param(
                'DevDataTrace3', 'buffer', np.array([[1155.0]], dtype=np.float32)
            ),
            pytest.param(
                'DevDataTrace4', 'buffer', np.array([[880.0]], dtype=np.float32)
            ),
            pytest.param(
                'DevDataTrace5', 'buffer', np.array([[90.0]], dtype=np.float32)
            ),
            pytest.param(
                'DevDataTrace6', 'buffer', np.array([[200.0]], dtype=np.float32)
            ),
            pytest.param(
                'DevDataTrace7', 'buffer', np.array([[60.0]], dtype=np.float32)
            ),
            pytest.param(
                'DevDataTrace8', 'buffer', np.array([[0.0]], dtype=np.float32)
            ),
            pytest.param(
                'DevDataTrace9', 'buffer', np.array([[400.85]], dtype=np.float32)
            ),
            pytest.param(
                'LoadFile',
                'buffer',
                Path(
                    'C:\\Users\\vs2418\\Repos\\lib\\davislib\\tests\\data\\SimpleImageSet\\B00001.im7'
                ),
            ),
            pytest.param(
                'LoadSet',
                'buffer',
                Path('C:/Users/vs2418/Repos/lib/davislib/tests/data/SimpleImageSet'),
            ),
            pytest.param('LoadSetIndex', 'buffer', '1'),
            pytest.param('Timestamp', 'buffer', '2025-02-12T11:55:25,594+01:00'),
            pytest.param('_DaVisVersion', 'buffer', '11.1.0.186'),
            pytest.param('_Date', 'buffer', '---'),
            pytest.param('_Header_PackType', 'buffer', '20'),
            pytest.param('_Time', 'buffer', '---'),
            pytest.param(
                'AOIused', 'frame', np.array([[0, 1067, 1, 1]], dtype=np.int32)
            ),
            pytest.param('Acq.AttributesTransformed', 'frame', '1'),
            pytest.param(
                'Acq.Camera.ConversionFactor', 'frame', np.array([[0.45776367]])
            ),
            pytest.param('Acq.Camera.ConversionFactor.Unit', 'frame', 'e-1/count'),
            pytest.param(
                'Acq.Camera.ID', 'frame', 'Camera.ImagerSCMOS5MCLHS: 61008958'
            ),
            pytest.param('Acq.Camera.Index', 'frame', np.array([[0]], dtype=np.int32)),
            pytest.param('Acq.Camera.Label', 'frame', 'Camera 1'),
            pytest.param('Acq.Camera.Noise', 'frame', np.array([[6.5536]])),
            pytest.param('Acq.Camera.Noise.Unit', 'frame', 'counts'),
            pytest.param('Acq.Camera.Spectrum', 'frame', 'visible'),
            pytest.param('Acq.Time', 'frame', np.array([[400.85]])),
            pytest.param('AcqTimeSeries', 'frame', '0.000 µs'),
            pytest.param('CCDExposureTime', 'frame', '9000 µs'),
            pytest.param('CamPixelSize', 'frame', '6.5 µm'),
            pytest.param('CameraMaxIntensity', 'frame', '65535'),
            pytest.param('CameraMaxNx', 'frame', '2560'),
            pytest.param('CameraMaxNy', 'frame', '2160'),
            pytest.param('CameraName', 'frame', '1: Imager sCMOS CLHS'),
            pytest.param('FrameProcessing', 'frame', '2'),
            pytest.param('FrameRotation', 'frame', '4'),
            pytest.param('RGBFrame', 'frame', '0'),
            pytest.param(
                'RealFrameSize', 'frame', np.array([[2560, 250]], dtype=np.int32)
            ),
        ],
    )
    def test_list_raw_attributes_and_check(self, data_path, key, level, value):
        with ImageSetAccessor(data_path / 'SimpleImageSet') as images:
            attrs = images.list_attributes(infer_types=False)
            assert len(attrs) == 114

            assert key in attrs
            assert attrs[key].level == AttributeLevel(level)

            actual_value = attrs[key].value
            if isinstance(value, np.ndarray) and isinstance(actual_value, np.ndarray):
                numpy.testing.assert_allclose(actual_value, value, strict=True)
            elif isinstance(value, Path) and (actual_value is not None):
                assert Path(actual_value) == value
            else:
                assert actual_value == value

            assert attrs[key].dimensions == images.dimensions

    @pytest.mark.parametrize(
        'key,level,dtype,shape,unit,value',
        [
            pytest.param(
                'DevDataTrace0',
                'buffer',
                np.float32,
                (),
                'µs',
                np.array(9000.005, dtype=np.float32),
            ),
            pytest.param(
                'DevDataTrace1',
                'buffer',
                np.float32,
                (),
                'nm',
                np.array(628.2353, dtype=np.float32),
            ),
            pytest.param(
                'DevDataTrace2',
                'buffer',
                np.float32,
                (),
                'counts',
                np.array(1557.0, dtype=np.float32),
            ),
            pytest.param(
                'DevDataTrace3',
                'buffer',
                np.float32,
                (),
                'counts',
                np.array(1155.0, dtype=np.float32),
            ),
            pytest.param(
                'DevDataTrace4',
                'buffer',
                np.float32,
                (),
                'ns',
                np.array(880.0, dtype=np.float32),
            ),
            pytest.param(
                'DevDataTrace5',
                'buffer',
                np.float32,
                (),
                '%',
                np.array(90.0, dtype=np.float32),
            ),
            pytest.param(
                'DevDataTrace6',
                'buffer',
                np.float32,
                (),
                'ns',
                np.array(200.0, dtype=np.float32),
            ),
            pytest.param(
                'DevDataTrace7',
                'buffer',
                np.float32,
                (),
                '%',
                np.array(60.0, dtype=np.float32),
            ),
            pytest.param(
                'DevDataTrace8',
                'buffer',
                np.float32,
                (),
                '%',
                np.array(0.0, dtype=np.float32),
            ),
            pytest.param(
                'DevDataTrace9',
                'buffer',
                np.float32,
                (),
                'µs',
                np.array(400.85, dtype=np.float32),
            ),
            pytest.param(
                'DevDataTrace10',
                'buffer',
                np.float32,
                (),
                'µs',
                np.array(0.0, dtype=np.float32),
            ),
            pytest.param(
                'RealFrameSize',
                'frame',
                np.int32,
                (2,),
                'dimensionless',
                np.array([2560, 250], dtype=np.int32),
            ),
            pytest.param('RGBFrame', 'frame', np.uint8, (), 'dimensionless', 0),
            pytest.param('FrameRotation', 'frame', np.uint8, (), 'dimensionless', 4),
            pytest.param('FrameProcessing', 'frame', np.uint8, (), 'dimensionless', 2),
            pytest.param(
                'CameraName',
                'frame',
                StringDType(),
                (),
                None,
                '1: Imager sCMOS CLHS',
            ),
            pytest.param('CameraMaxNy', 'frame', np.uint16, (), 'dimensionless', 2160),
            pytest.param('CameraMaxNx', 'frame', np.uint16, (), 'dimensionless', 2560),
            pytest.param(
                'CameraMaxIntensity', 'frame', np.uint16, (), 'dimensionless', 65535
            ),
            pytest.param('CamPixelSize', 'frame', np.float16, (), 'micrometer', 6.5),
            pytest.param(
                'CCDExposureTime', 'frame', np.uint16, (), 'microsecond', 9000
            ),
            pytest.param('AcqTimeSeries', 'frame', np.float16, (), 'microsecond', 0.0),
            pytest.param(
                'Acq.Time', 'frame', np.float64, (), 'dimensionless', np.array(400.85)
            ),
            pytest.param(
                'Acq.Camera.Spectrum', 'frame', StringDType(), (), None, 'visible'
            ),
            pytest.param(
                'Acq.Camera.Noise', 'frame', np.float64, (), 'counts', np.array(6.5536)
            ),
            pytest.param(
                'Acq.Camera.Label', 'frame', StringDType(), (), None, 'Camera 1'
            ),
            pytest.param(
                'Acq.Camera.Index',
                'frame',
                np.int32,
                (),
                'dimensionless',
                np.array(0, dtype=np.int32),
            ),
            pytest.param(
                'Acq.Camera.ID',
                'frame',
                StringDType(),
                (),
                None,
                'Camera.ImagerSCMOS5MCLHS: 61008958',
            ),
            pytest.param(
                'Acq.Camera.ConversionFactor',
                'frame',
                np.float64,
                (),
                'e-1/count',
                np.array(0.45776367),
            ),
            pytest.param(
                'Acq.AttributesTransformed', 'frame', np.uint8, (), 'dimensionless', 1
            ),
            pytest.param(
                'AOIused',
                'frame',
                np.int32,
                (4,),
                'dimensionless',
                np.array([0, 1067, 1, 1], dtype=np.int32),
            ),
            pytest.param('_Time', 'buffer', StringDType(), (), None, '---'),
            pytest.param(
                '_Header_PackType', 'buffer', np.uint8, (), 'dimensionless', 20
            ),
            pytest.param('_Date', 'buffer', StringDType(), (), None, '---'),
            pytest.param(
                '_DaVisVersion', 'buffer', StringDType(), (), None, '11.1.0.186'
            ),
            pytest.param(
                'Timestamp',
                'buffer',
                np.dtype('datetime64[us]'),
                (),
                None,
                np.datetime64('2025-02-12T11:55:25.594000'),
            ),
            pytest.param('LoadSetIndex', 'buffer', np.uint8, (), 'dimensionless', 1),
            pytest.param(
                'LoadSet',
                'buffer',
                StringDType(),
                (),
                None,
                Path('C:/Users/vs2418/Repos/lib/davislib/tests/data/SimpleImageSet'),
            ),
            pytest.param(
                'LoadFile',
                'buffer',
                StringDType(),
                (),
                None,
                Path(
                    'C:\\Users\\vs2418\\Repos\\lib\\davislib\\tests\\data\\SimpleImageSet\\B00001.im7'
                ),
            ),
            pytest.param(
                'CustomImageTag_Count', 'buffer', np.uint8, (), 'dimensionless', 0
            ),
            pytest.param(
                'Acq.Status.RecordPost', 'buffer', np.uint8, (), 'dimensionless', 0
            ),
            pytest.param(
                'Acq.Input.StartTrigger', 'buffer', np.uint8, (), 'dimensionless', 0
            ),
            pytest.param(
                'Acq.Input.SpeedSelect', 'buffer', np.uint8, (), 'dimensionless', 0
            ),
        ],
    )
    def test_list_attributes(self, data_path, key, level, dtype, shape, unit, value):
        with ImageSetAccessor(data_path / 'SimpleImageSet') as images:
            attrs = images.list_attributes()
            assert len(attrs) == 43

            assert key in attrs
            assert attrs[key].level == AttributeLevel(level)
            assert attrs[key].dtype == dtype
            assert attrs[key].shape == shape
            if unit is None:
                assert attrs[key].unit is None
            else:
                assert attrs[key].unit == unit

            actual_value = attrs[key].value
            if isinstance(value, np.ndarray) and isinstance(actual_value, np.ndarray):
                numpy.testing.assert_allclose(actual_value, value, strict=True)
            elif isinstance(value, Path) and (actual_value is not None):
                assert Path(actual_value) == value
            else:
                assert actual_value == value

            assert len(attrs[key].dimensions) == len(images.dimensions) + len(shape)
            assert attrs[key].dimensions == images.dimensions.with_dimensions(
                **{f'dim_{i}': size for i, size in enumerate(shape)}
            )


class TestGetData:

    def test_single_buffer(self, data_path):
        with ImageSetAccessor(data_path / 'SimpleImageSet') as images:
            data = images.get_data('PIXEL', buffer=0, y=slice(None), x=slice(None))
            assert data.shape == (250, 2560)
            assert data.dtype == np.uint16

    def test_single_buffer_squeeze_false(self, data_path):
        with ImageSetAccessor(data_path / 'SimpleImageSet', squeeze=False) as images:
            data = images.get_data(
                'PIXEL',
                buffer=0,
                frame=0,
                nz=slice(None),
                ny=slice(None),
                nx=slice(None),
            )
            assert data.shape == (1, 1, 1, 250, 2560)
            assert data.dtype == np.uint16

    def test_single_pixel(self, data_path):
        with ImageSetAccessor(data_path / 'SimpleImageSet') as images:
            data = images.get_data('PIXEL', buffer=slice(None), y=100, x=100)
            assert data.shape == (10,)
            assert list(data) == [
                10087,
                10269,
                10549,
                10411,
                11807,
                9497,
                10615,
                10612,
                10690,
                10058,
            ]

    def test_single_column(self, data_path):
        with ImageSetAccessor(data_path / 'SimpleImageSet') as images:
            data = images.get_data('PIXEL', buffer=0, y=slice(None), x=100)
            assert data.shape == (250,)
            numpy.testing.assert_array_equal(
                data,
                # fmt: off
                np.array(
                    [
                        602, 607, 595, 537, 613, 560, 577, 538, 607, 621, 595, 605, 607,  # noqa
                        550, 643, 622, 576, 647, 607, 663, 672, 671, 679, 647, 605, 633,  # noqa
                        693, 692, 662, 693, 649, 721, 714, 647, 591, 676, 645, 664, 632,  # noqa
                        694, 690, 661, 712, 636, 668, 660, 617, 748, 745, 748, 729, 772,  # noqa
                        758, 722, 771, 730, 819, 804, 804, 811, 848, 888, 920, 932, 1002,  # noqa
                        948, 998, 1160, 1197, 1208, 1214, 1180, 1371, 1351, 1436, 1460,  # noqa
                        1501, 1691, 1731, 1941, 2082, 2327, 2554, 2719, 3159, 3606, 3982,  # noqa
                        4311, 4900, 5384, 5923, 6612, 7113, 7619, 8101, 8411, 8875, 8875,  # noqa
                        9540, 9313, 10087, 10024, 10192, 10574, 10353, 10256, 10710,  # noqa
                        11136, 11014, 10781, 11111, 10877, 11178, 11297, 11221, 11268,  # noqa
                        11584, 11836, 11542, 11788, 12285, 12037, 12461, 12375, 12550,  # noqa
                        13001, 13043, 13024, 13589, 13745, 13223, 13135, 13350, 13434,  # noqa
                        13234, 13266, 13109, 13242, 13033, 13125, 13333, 13178, 12991,  # noqa
                        13102, 12560, 12477, 12296, 11971, 11590, 11781, 11687, 11194,  # noqa
                        10985, 10434, 10152, 10138, 10291, 9725, 9467, 8817, 8589, 7981,  # noqa
                        7191, 6600, 5968, 4968, 4458, 3660, 3190, 2727, 2428, 2069, 1895,  # noqa
                        1710, 1531, 1625, 1483, 1359, 1374, 1252, 1233, 1220, 1079, 1020,  # noqa
                        1103, 964, 921, 922, 884, 825, 774, 801, 774, 721, 784, 714, 717,  # noqa
                        716, 716, 773, 721, 745, 728, 729, 775, 707, 794, 787, 756, 812,  # noqa
                        828, 808, 862, 818, 840, 850, 790, 793, 765, 764, 732, 729, 737,  # noqa
                        654, 687, 672, 605, 637, 617, 642, 657, 660, 629, 644, 720, 756,  # noqa
                        992, 1292, 1044, 783, 754, 704, 693, 698, 630, 661, 736, 708, 680,  # noqa
                        757,
                    ],
                    dtype=np.uint16,
                ),
                # fmt: on
            )

    def test_get_timestamp_attribute(self, data_path):
        with ImageSetAccessor(data_path / 'SimpleImageSet') as images:
            timestamp = images.get_attribute('Timestamp')

            expected = np.array(
                [
                    '2025-02-12T11:55:25.594000',
                    '2025-02-12T11:55:25.694000',
                    '2025-02-12T11:55:25.794000',
                    '2025-02-12T11:55:25.894000',
                    '2025-02-12T11:55:25.994000',
                    '2025-02-12T11:55:26.094000',
                    '2025-02-12T11:55:26.194000',
                    '2025-02-12T11:55:26.294000',
                    '2025-02-12T11:55:26.394000',
                    '2025-02-12T11:55:26.494000',
                ],
                dtype='datetime64[us]',
            )

            numpy.testing.assert_array_equal(timestamp, expected, strict=True)

    def test_get_timestamp_attribute_for_single_buffer(self, data_path):
        with ImageSetAccessor(data_path / 'SimpleImageSet') as images:
            timestamp = images.get_attribute('Timestamp', buffer=2)

            expected = np.array(
                '2025-02-12T11:55:25.794000',
                dtype='datetime64[us]',
            )

            numpy.testing.assert_array_equal(timestamp, expected, strict=True)

    def test_get_timestamp_attribute_for_slice(self, data_path):
        with ImageSetAccessor(data_path / 'SimpleImageSet') as images:
            timestamp = images.get_attribute('Timestamp', buffer=slice(2, 5))

            expected = np.array(
                [
                    '2025-02-12T11:55:25.794000',
                    '2025-02-12T11:55:25.894000',
                    '2025-02-12T11:55:25.994000',
                ],
                dtype='datetime64[us]',
            )

            numpy.testing.assert_array_equal(timestamp, expected, strict=True)

    def test_multidimensional_attribute(self, data_path):
        with ImageSetAccessor(data_path / 'SimpleImageSet') as images:
            timestamp = images.get_attribute('AOIused')

            expected = np.array(
                [
                    [0, 1067, 1, 1],
                    [0, 1067, 1, 1],
                    [0, 1067, 1, 1],
                    [0, 1067, 1, 1],
                    [0, 1067, 1, 1],
                    [0, 1067, 1, 1],
                    [0, 1067, 1, 1],
                    [0, 1067, 1, 1],
                    [0, 1067, 1, 1],
                    [0, 1067, 1, 1],
                ],
                dtype=np.int32,
            )

            numpy.testing.assert_array_equal(timestamp, expected, strict=True)

    def test_multidimensional_attribute_with_slicing_attribute_data(self, data_path):
        with ImageSetAccessor(data_path / 'SimpleImageSet') as images:
            timestamp = images.get_attribute(
                'AOIused', buffer=slice(None), dim_0=slice(1, 3)
            )

            expected = np.array(
                [
                    [1067, 1],
                    [1067, 1],
                    [1067, 1],
                    [1067, 1],
                    [1067, 1],
                    [1067, 1],
                    [1067, 1],
                    [1067, 1],
                    [1067, 1],
                    [1067, 1],
                ],
                dtype=np.int32,
            )

            numpy.testing.assert_array_equal(timestamp, expected, strict=True)
