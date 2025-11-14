# hypercnn/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# --- HyperLogBlock (v1 Mejorado) ---
class HyperLogBlock(nn.Module):
    """
    Bloque convolucional con conexiones hiperbólicas y manejo robusto de dimensiones.
    """
    def __init__(self, in_ch, out_ch, stride=1, is_first=False, activation='silu'):
        super().__init__()
        self.is_first = is_first
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.stride = stride
        self.activation_name = activation

        # Seleccionar función de activación
        if activation == 'silu':
            act_fn = nn.SiLU(inplace=True)
        elif activation == 'leaky_relu':
            act_fn = nn.LeakyReLU(0.1, inplace=True)
        else: # 'relu' u otros
            act_fn = nn.ReLU(inplace=True)

        # Ramas convolucionales
        self.branch3x3 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            act_fn
        )
        self.branch1x1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
            nn.BatchNorm2d(out_ch),
            act_fn
        )

        # Conexión identidad
        self.use_identity = (stride == 1 and in_ch == out_ch)
        if self.use_identity:
            self.identity = nn.Identity()
        else:
            self.identity = None

        # Proyección para cambios de dimensión
        if stride != 1 or in_ch != out_ch:
            self.main_proj = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )
            # Inicialización específica para proyección principal
            self._init_submodule_weights(self.main_proj, activation)
        else:
            self.main_proj = None

        # Diccionario para proyecciones dinámicas de vecinos hiperbólicos
        self.hyper_proj = nn.ModuleDict()
        # Buffer para la convolución de fusión (creada dinámicamente si es necesario)
        self.fusion_conv = None
        # Atributo para rastrear el dispositivo
        self._device = None

    def _init_submodule_weights(self, module, activation_name):
        """Inicialización específica para submódulos como proyecciones."""
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                nonlinearity = activation_name
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity=nonlinearity)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def _get_or_create_projection(self, in_channels, out_channels, spatial_match=False, target_device=None):
        """Crear o recuperar una proyección para dimensiones específicas."""
        device = target_device or self._device
        key = f"{in_channels}_{out_channels}_{1 if spatial_match else 0}"

        if key not in self.hyper_proj:
            layers = []
            # Proyección de canales
            conv_layer = nn.Conv2d(in_channels, out_channels, 1, bias=False)
            bn_layer = nn.BatchNorm2d(out_channels)
            layers.extend([conv_layer, bn_layer])

            # Ajuste espacial si es necesario (Upsampling)
            if not spatial_match:
                layers.append(nn.Upsample(scale_factor=2, mode='nearest'))

            # Crear secuencia y mover al dispositivo correcto si se proporciona
            proj_module = nn.Sequential(*layers)
            if device is not None:
                proj_module = proj_module.to(device)

            # Inicializar pesos de la proyección
            self._init_submodule_weights(proj_module, self.activation_name)

            self.hyper_proj[key] = proj_module

        # Asegurar que el módulo esté en el dispositivo correcto antes de retornarlo
        proj = self.hyper_proj[key]
        if device is not None and next(proj.parameters()).device != device:
             proj = proj.to(device)
             self.hyper_proj[key] = proj # Actualizar en el dict

        return proj

    def forward(self, x, neighbors):
        # Actualizar dispositivo interno si es la primera llamada
        if self._device is None:
            self._device = x.device

        # Procesar conexiones hiperbólicas
        if not self.is_first and neighbors:
            adjusted_neighbors = []
            for n in neighbors:
                # Asegurar que el vecino esté en el mismo dispositivo
                if n.device != x.device:
                    n = n.to(x.device)

                # Caso 1: Mismas dimensiones espaciales y de canales
                if n.shape[2:] == x.shape[2:] and n.shape[1] == x.shape[1]:
                    adjusted_neighbors.append(n)
                # Caso 2: Mismas dimensiones espaciales pero diferentes canales
                elif n.shape[2:] == x.shape[2:] and n.shape[1] != x.shape[1]:
                    proj = self._get_or_create_projection(n.shape[1], x.shape[1], spatial_match=True, target_device=x.device)
                    adjusted_neighbors.append(proj(n))
                # Caso 3: Diferentes dimensiones espaciales y/o de canales
                else:
                    # Ajustar dimensiones espaciales si es necesario
                    if n.shape[2:] != x.shape[2:]:
                        n = F.interpolate(n, size=x.shape[2:], mode='nearest')
                    # Luego ajustar canales
                    if n.shape[1] != x.shape[1]:
                        proj = self._get_or_create_projection(n.shape[1], x.shape[1], spatial_match=True, target_device=x.device)
                        n = proj(n)
                    adjusted_neighbors.append(n)

            # *** CORRECCIÓN CLAVE ***
            # Usar concatenación + convolución de fusión en lugar de torch.stack
            if adjusted_neighbors:
                all_inputs = [x] + adjusted_neighbors
                concat_inputs = torch.cat(all_inputs, dim=1)

                # Crear o reutilizar la convolución de fusión
                if self.fusion_conv is None or self.fusion_conv[0].weight.device != x.device or self.fusion_conv[0].in_channels != concat_inputs.shape[1]:
                    self.fusion_conv = nn.Sequential(
                        nn.Conv2d(concat_inputs.shape[1], x.shape[1], 1, bias=False),
                        nn.BatchNorm2d(x.shape[1]),
                        nn.SiLU(inplace=True) if self.activation_name == 'silu' else nn.ReLU(inplace=True)
                    ).to(x.device)
                    # Inicializar la convolución de fusión
                    self._init_submodule_weights(self.fusion_conv, self.activation_name)

                agg = self.fusion_conv(concat_inputs)
            else:
                agg = x
        else:
            agg = x

        # Procesar ramas convolucionales
        out3x3 = self.branch3x3(agg)
        out1x1 = self.branch1x1(agg)
        out = out3x3 + out1x1

        # Conexión residual
        if self.use_identity:
            out = out + self.identity(agg)
        elif self.main_proj is not None: # Si no hay identidad, usar proyección principal
            res = self.main_proj(agg if self.stride != 1 or self.in_ch != self.out_ch else x)
            out = out + res
        # Si no hay ni identidad ni proyección principal, no se añade residuo (caso raro)

        return out

  # HyperCNN 
class HyperCNN(nn.Module):
    """
    HyperCNN v1: Arquitectura original con mejoras en inicialización y manejo de dispositivos.
    """
    def __init__(self, num_classes=10, depth=16, base_channels=32, activation='silu'):
        super().__init__()
        self.depth = depth
        self.activation = activation

        # Generar identificadores hiperbólicos (base-4)
        self.num_digits = max(1, math.ceil(math.log(depth, 4)))
        self.block_ids = [self._to_hyper_id(i) for i in range(depth)]

        # Precomputar conexiones hiperbólicas
        self.connections = self._build_hyper_connections()

        # Stem inicial
        self.stem = nn.Sequential(
            nn.Conv2d(3, base_channels, kernel_size=4, stride=4),
            nn.BatchNorm2d(base_channels),
            nn.SiLU(inplace=True) if activation=='silu' else nn.ReLU(inplace=True)
        )
        # Inicializar Stem
        self._init_submodule_weights(self.stem, activation)

        # Crear bloques convolucionales
        self.blocks = nn.ModuleList()
        current_ch = base_channels

        for i in range(depth):
            # Aumentar canales cada 8 bloques
            if i > 0 and (i % 8 == 0):
                next_ch = current_ch * 2
                stride = 2 if i < depth - 4 else 1  # No reducir al final
            else:
                next_ch = current_ch
                stride = 1 if i == 0 else 1

            # Crear bloque
            block = HyperLogBlock(
                current_ch,
                next_ch,
                stride=stride,
                is_first=(i == 0),
                activation=activation
            )
            # El bloque se auto-inicializa en su __init__
            self.blocks.append(block)
            current_ch = next_ch

        # Cabeza de clasificación
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(current_ch, num_classes)
        )
        # Inicializar Head
        self._init_head_weights()

    def _init_submodule_weights(self, module, activation_name):
        """Inicialización específica para módulos como stem/head."""
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                nonlinearity = activation_name
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity=nonlinearity)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01) # Std típica para capa final
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _init_head_weights(self):
        """Inicialización específica para la cabeza final."""
        # La capa lineal final suele inicializarse con normal(0, 0.01)
        final_linear = self.head[-1] # Asume que la última capa es nn.Linear
        if isinstance(final_linear, nn.Linear):
             nn.init.normal_(final_linear.weight, 0, 0.01)
             if final_linear.bias is not None:
                 nn.init.constant_(final_linear.bias, 0)

    def _to_hyper_id(self, idx):
        """Convertir índice a identificador hiperbólico (base-4)"""
        digits = []
        for _ in range(self.num_digits):
            digits.append(idx % 4)
            idx //= 4
        return digits

    def _build_hyper_connections(self):
        """Precomputar conexiones basadas en coincidencia de dígitos"""
        connections = {}

        for i in range(self.depth):
            neighbors = []
            for j in range(i):  # Solo conexiones hacia atrás
                # Conectar si coinciden en AL MENOS UN dígito en la MISMA POSICIÓN
                if any(self.block_ids[i][k] == self.block_ids[j][k] for k in range(self.num_digits)):
                    neighbors.append(j)

            # Garantizar mínimo de conexiones
            if len(neighbors) < 2 and i > 0:
                neighbors.extend([max(0, i-2), max(0, i-4)])
                neighbors = list(set(neighbors))  # Eliminar duplicados

            connections[i] = neighbors

        return connections

    def forward(self, x):
        # Stem inicial
        x = self.stem(x)

        # Estados intermedios para conexiones hiperbólicas
        states = [None] * self.depth

        # Procesar cada bloque
        for i, block in enumerate(self.blocks):
            # Obtener vecinos válidos
            neighbors = []
            for j in self.connections[i]:
                if j < i and states[j] is not None:
                    # Asegurar que el vecino esté en el mismo dispositivo que x
                    if states[j].device != x.device:
                        states[j] = states[j].to(x.device)
                    neighbors.append(states[j])

            if i == 0:
                states[0] = block(x, neighbors)
            else:
                # Asegurar que el estado anterior esté en el dispositivo correcto
                if states[i-1].device != x.device:
                     states[i-1] = states[i-1].to(x.device)
                states[i] = block(states[i-1], neighbors)

        # Clasificación final
        return self.head(states[-1])

    def _apply(self, fn):
        """Hook para asegurar que módulos dinámicos se muevan al dispositivo correcto."""
        super()._apply(fn)
        # Actualizar dispositivo interno de los bloques si se mueve el modelo
        for block in self.blocks:
            if hasattr(block, '_device') and block._device is not None:
                block._device = fn(block._device) if hasattr(fn, '__self__') and hasattr(fn.__self__, 'type_as') else block._device
        return self

