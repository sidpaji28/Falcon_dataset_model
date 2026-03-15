export default function LicensingDashboard() {
  return (
    <div className="p-8">
      <h1 className="text-3xl font-bold mb-4">Licensing Dashboard</h1>
      <p>Manage enterprise seats and token consumption limits for the AI agent features.</p>
      <div className="mt-4 p-4 border rounded">
        <h2 className="text-xl font-semibold">Gemini API Token Consumption</h2>
        <div className="w-full bg-gray-200 rounded-full h-2.5 dark:bg-gray-700 mt-2">
          <div className="bg-blue-600 h-2.5 rounded-full" style={{ width: '45%' }}></div>
        </div>
        <p className="mt-2 text-sm text-gray-500">45% of monthly quota used.</p>
      </div>
    </div>
  );
}
