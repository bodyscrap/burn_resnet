use burn::{
    module::Module,
    nn::{
        conv::{Conv2d, Conv2dConfig},
        pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig, MaxPool2d, MaxPool2dConfig},
        BatchNorm, BatchNormConfig, Linear, LinearConfig, PaddingConfig2d, Relu,
    },
    tensor::{backend::Backend, Tensor},
};

#[derive(Module, Debug)]
struct DownsampleBlock<B: Backend> {
    conv: Conv2d<B>,
    bn: BatchNorm<B, 2>,
}

#[derive(Module, Debug)]
pub struct Bottleneck<B: Backend> {
    conv1: Conv2d<B>,
    bn1: BatchNorm<B, 2>,
    conv2: Conv2d<B>,
    bn2: BatchNorm<B, 2>,
    conv3: Conv2d<B>,
    bn3: BatchNorm<B, 2>,
    downsample: Option<DownsampleBlock<B>>,
    relu: Relu,
}

impl<B: Backend> Bottleneck<B> {
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let identity = x.clone();

        let out = self.conv1.forward(x);
        let out = self.bn1.forward(out);
        let out = self.relu.forward(out);

        let out = self.conv2.forward(out);
        let out = self.bn2.forward(out);
        let out = self.relu.forward(out);

        let out = self.conv3.forward(out);
        let out = self.bn3.forward(out);

        let out = if let Some(ref downsample) = self.downsample {
            let identity = downsample.conv.forward(identity);
            let identity = downsample.bn.forward(identity);
            out + identity
        } else {
            out + identity
        };

        self.relu.forward(out)
    }
}

#[derive(Module, Debug)]
pub struct ResNet<B: Backend> {
    conv1: Conv2d<B>,
    bn1: BatchNorm<B, 2>,
    relu: Relu,
    maxpool: MaxPool2d,
    layer1: Vec<Bottleneck<B>>,
    layer2: Vec<Bottleneck<B>>,
    layer3: Vec<Bottleneck<B>>,
    layer4: Vec<Bottleneck<B>>,
    avgpool: AdaptiveAvgPool2d,
    fc: Linear<B>,
}

impl<B: Backend> ResNet<B> {
    pub fn resnet50(num_classes: usize, device: &B::Device) -> Self {
        let conv1 = Conv2dConfig::new([3, 64], [7, 7])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(3, 3))
            .with_bias(false)
            .init(device);

        let bn1 = BatchNormConfig::new(64).init(device);
        let relu = Relu::new();
        let maxpool = MaxPool2dConfig::new([3, 3])
            .with_strides([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init();

        let layer1 = Self::make_layer(64, 64, 256, 3, 1, device);
        let layer2 = Self::make_layer(256, 128, 512, 4, 2, device);
        let layer3 = Self::make_layer(512, 256, 1024, 6, 2, device);
        let layer4 = Self::make_layer(1024, 512, 2048, 3, 2, device);

        let avgpool = AdaptiveAvgPool2dConfig::new([1, 1]).init();
        let fc = LinearConfig::new(2048, num_classes).init(device);

        Self {
            conv1,
            bn1,
            relu,
            maxpool,
            layer1,
            layer2,
            layer3,
            layer4,
            avgpool,
            fc,
        }
    }

    fn make_layer(
        inplanes: usize,
        planes: usize,
        out_planes: usize,
        blocks: usize,
        stride: usize,
        device: &B::Device,
    ) -> Vec<Bottleneck<B>> {
        let mut layers = Vec::new();

        // First block with potential downsampling
        let downsample = if stride != 1 || inplanes != out_planes {
            Some(DownsampleBlock {
                conv: Conv2dConfig::new([inplanes, out_planes], [1, 1])
                    .with_stride([stride, stride])
                    .with_bias(false)
                    .init(device),
                bn: BatchNormConfig::new(out_planes).init(device),
            })
        } else {
            None
        };

        layers.push(Bottleneck {
            conv1: Conv2dConfig::new([inplanes, planes], [1, 1])
                .with_bias(false)
                .init(device),
            bn1: BatchNormConfig::new(planes).init(device),
            conv2: Conv2dConfig::new([planes, planes], [3, 3])
                .with_stride([stride, stride])
                .with_padding(PaddingConfig2d::Explicit(1, 1))
                .with_bias(false)
                .init(device),
            bn2: BatchNormConfig::new(planes).init(device),
            conv3: Conv2dConfig::new([planes, out_planes], [1, 1])
                .with_bias(false)
                .init(device),
            bn3: BatchNormConfig::new(out_planes).init(device),
            downsample,
            relu: Relu::new(),
        });

        // Remaining blocks
        for _ in 1..blocks {
            layers.push(Bottleneck {
                conv1: Conv2dConfig::new([out_planes, planes], [1, 1])
                    .with_bias(false)
                    .init(device),
                bn1: BatchNormConfig::new(planes).init(device),
                conv2: Conv2dConfig::new([planes, planes], [3, 3])
                    .with_padding(PaddingConfig2d::Explicit(1, 1))
                    .with_bias(false)
                    .init(device),
                bn2: BatchNormConfig::new(planes).init(device),
                conv3: Conv2dConfig::new([planes, out_planes], [1, 1])
                    .with_bias(false)
                    .init(device),
                bn3: BatchNormConfig::new(out_planes).init(device),
                downsample: None,
                relu: Relu::new(),
            });
        }

        layers
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 2> {
        let mut x = self.conv1.forward(x);
        x = self.bn1.forward(x);
        x = self.relu.forward(x);
        x = self.maxpool.forward(x);

        for block in &self.layer1 {
            x = block.forward(x);
        }
        for block in &self.layer2 {
            x = block.forward(x);
        }
        for block in &self.layer3 {
            x = block.forward(x);
        }
        for block in &self.layer4 {
            x = block.forward(x);
        }

        x = self.avgpool.forward(x);
        let [batch_size, channels, _, _] = x.dims();
        let x = x.reshape([batch_size, channels]);
        self.fc.forward(x)
    }
    
    /// 最終層（分類ヘッド）を新しいクラス数用に置き換える
    #[allow(dead_code)]
    pub fn replace_head(mut self, num_classes: usize, device: &B::Device) -> Self {
        self.fc = LinearConfig::new(2048, num_classes).init(device);
        self
    }
}
