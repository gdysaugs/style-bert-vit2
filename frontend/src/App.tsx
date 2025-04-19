import React, { useState, useEffect } from 'react';

// バックエンドAPIのエンドポイント
const BACKEND_API_URL = import.meta.env.VITE_BACKEND_API_URL || 'http://localhost:8000'; // 環境変数から取得、なければデフォルト値

function App() {
  const [text, setText] = useState<string>('');
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [prevAudioUrl, setPrevAudioUrl] = useState<string | null>(null); // 前回のURLを保持

  // audioUrl が更新されたら、古い Blob URL を解放する
  useEffect(() => {
    if (prevAudioUrl) {
      URL.revokeObjectURL(prevAudioUrl);
    }
    // 現在の audioUrl を次回の解放のために保持
    setPrevAudioUrl(audioUrl);

    // コンポーネントアンマウント時に最後のURLを解放 (念のため)
    return () => {
      if (audioUrl) {
        URL.revokeObjectURL(audioUrl);
      }
    };
    // audioUrl が変更されたときだけ実行
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [audioUrl]);

  const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault(); // フォームのデフォルト送信を抑制
    setIsLoading(true);
    setError(null);
    // setAudioUrl(null); // ここではクリアせず、新しいものができたら更新する

    if (!text.trim()) {
      setError('テキストを入力してください。');
      setIsLoading(false);
      return;
    }

    try {
      const response = await fetch(`${BACKEND_API_URL}/synthesize`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text: text }), // 入力テキストをJSONで送信
      });

      if (!response.ok) {
        // HTTPステータスが 2xx 以外の場合
        let errorDetail = `HTTPエラー: ${response.status}`;
        try {
          // エラーレスポンスのボディに詳細が含まれているか試す
          const errorData = await response.json();
          errorDetail = errorData.detail || JSON.stringify(errorData);
        } catch (e) {
          // JSONとしてパースできなかった場合
          errorDetail = `${errorDetail} - ${response.statusText}`;
        }
        throw new Error(errorDetail);
      }

      // レスポンスボディから音声データをBlobとして取得
      const audioBlob = await response.blob();

      // 新しい Blob URL を作成して state にセット
      const newUrl = URL.createObjectURL(audioBlob);
      setAudioUrl(newUrl);

    } catch (err: any) {
      console.error('音声合成リクエストエラー:', err);
      setError(err.message || '音声の生成中にエラーが発生しました。');
      setAudioUrl(null); // エラー時はクリア
    } finally {
      setIsLoading(false);
    }
  };


  return (
    <div className="min-h-screen bg-gray-100 flex flex-col items-center justify-center p-4">
      <h1 className="text-3xl font-bold text-blue-600 mb-8">
        Style-Bert-VITS2 チャット風アプリ
      </h1>

      <form onSubmit={handleSubmit} className="w-full max-w-lg bg-white p-8 rounded-lg shadow-md">
        <div className="mb-4">
          <label htmlFor="text-input" className="block text-gray-700 text-sm font-bold mb-2">
            読み上げたいテキストを入力:
          </label>
          <textarea
            id="text-input"
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="ここに日本語テキストを入力..."
            rows={4}
            className="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:shadow-outline"
            disabled={isLoading}
          />
        </div>

        <div className="flex items-center justify-between flex-wrap"> {/* 折り返し対応 */}
          <button
            type="submit"
            className={`bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded focus:outline-none focus:shadow-outline mb-2 sm:mb-0 ${isLoading ? 'opacity-50 cursor-not-allowed' : ''}`} // レスポンシブ対応
            disabled={isLoading}
          >
            {isLoading ? '生成中...' : '音声生成'}
          </button>

          {/* 音声再生エリア */}
          {audioUrl && !isLoading && (
             <audio controls autoPlay src={audioUrl} className="ml-0 sm:ml-4 flex-grow"> {/* autoPlay 追加、レスポンシブ対応 */}
               お使いのブラウザはオーディオ再生に対応していません。
             </audio>
           )}
        </div>

        {/* 処理状態・エラー表示エリア */}
        <div className="mt-4 h-6"> {/* 高さを固定してレイアウトが崩れないように */}
          {isLoading && (
            <p className="text-sm text-gray-600">音声を生成しています...</p>
          )}
          {error && (
            <p className="text-sm text-red-500">エラー: {error}</p>
          )}
          {!isLoading && !error && audioUrl && (
            <p className="text-sm text-green-600">音声が生成されました。</p> // 成功メッセージ追加
          )}
        </div>

      </form>

      <footer className="mt-8 text-sm text-gray-500">
        <p>Backend API: {BACKEND_API_URL}</p>
      </footer>
    </div>
  );
}

export default App; 