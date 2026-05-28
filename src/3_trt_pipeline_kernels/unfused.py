import os
import onnx
import onnx_graphsurgeon as gs

# 1. 경로 설정
# 현재 실행 중인 스크립트의 절대 경로 (sy 폴더 안)
current_dir = os.path.dirname(os.path.abspath(__file__))

# 읽어올 원본 파일은 sparse 모델로 지정
onnx_path = os.path.abspath(os.path.join(current_dir, "..", "sunghyun", "deit_onnx", "deit_small_sparse.onnx"))

# 저장할 파일명도 구분하기 쉽게 sparse_unfused 등으로 변경
save_path = os.path.abspath(os.path.join(current_dir, "..", "sunghyun", "deit_onnx", "deit_small_sparse_unfused.onnx"))

print(f"읽어올 경로: {onnx_path}")

# 2. 기존 ONNX 모델 로드
graph = gs.import_onnx(onnx.load(onnx_path))
            
        )
        
        identity_node = gs.Node(op="Identity", inputs=[matmul_out], outputs=[identity_out])
        graph.nodes.append(identity_node)
        
        for next_node in graph.nodes:
            if next_node != identity_node and matmul_out in next_node.inputs:
                for i, inp in enumerate(next_node.inputs):
                    if inp == matmul_out:
                        next_node.inputs[i] = identity_out
                        print(f"{next_node.name}의 입력을 Identity 노드로 연결 변경 완료")

# 4. 그래프 정리 및 저장
graph.cleanup().toposort()
onnx.save(gs.export_onnx(graph), save_path)
print(f"저장 완료: {save_path}")